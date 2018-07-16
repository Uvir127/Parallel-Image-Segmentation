// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include <mpi.h>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

const char *imageFilename = "lena_bw_double.pgm";

double overhead, overheadS, overheadE = 0;

void calcHist(float *iData, float *hist, float *sumB, float *wB, int width, int height, int from, int till, int rank){
  float output[256];
  memset(output, 0, 256*sizeof(float));

  for(int i = from; i <= till; i++){
    int h = round(iData[i]*255);
    output[h] += 1;
  }

  overheadS = MPI_Wtime();
  MPI_Allreduce(&output, hist, 256, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  overheadE = MPI_Wtime();
  overhead += overheadE - overheadS;
  
  if(rank == 0){ 
    std::partial_sum (hist, hist+256, wB);
    
    float temp = 0;
    for (int j = 0; j < 256; j++) {
      temp += j * hist[j];
      sumB[j] = temp;
    }
  }
}


void otsu(int from, int till, float* hist, float *sumB, float *wB, float *thold, float *vars, int width, int height){
  float sum = sumB[255];
  int total = width*height;
  float wF = 0;

  float maxVar = 0;

  for (int t = from ; t <= till ; t++) {
    wF = total - wB[t];             // Weight Foreground

    if (wB[t] != 0 || wF != 0){
      float mF = (sum - sumB[t]) / wF;    // Mean Foreground
      float mB = sumB[t] / wB[t];            // Mean Background

      // Calculate Between Class Variance
      float varB = wB[t] * wF * (mB - mF) * (mB - mF);
      if(isnan(varB))
        varB = 0;

      // Check if new maximum found
      if (varB > maxVar) {
        maxVar = varB;
        thold[0] = t;
      }
    }
  }
  overheadS = MPI_Wtime();
  MPI_Gather(&maxVar, 1, MPI_FLOAT, vars, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  overheadE = MPI_Wtime();
  overhead += overheadE - overheadS;
}

int findMax(float *array, int N){
  int k = 0;
  float max = array[k];
  for (int i = 1; i < N; i++){
    if (array[i] > max){
      max = array[i];
      k = i;
    }
  }
  return k;
}

void threshold(float *iData, float *oData, float *thold, int width, int height){
  int total = width*height;
  for (size_t i = 0; i < total; i++) {
    float val = iData[i]*255;
    if(val > thold[0])
      oData[i] = 1;
  }
}

int main(int argc, char *argv[]){
  int nprocess, rank = 1;
  double start, end = 0;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  float *hData = NULL;
  char outputFilename[1024];

  sdkLoadPGM(imagePath, &hData, &width, &height);

  size_t size = (width * height) * sizeof(float);
  size_t histSize = 256*sizeof(float);

  float *hOutputData = (float *)malloc(size);
  float *hist = (float *)malloc(histSize);
  float *wB = (float *)malloc(histSize);
  float *sumB = (float *)malloc(histSize);
  float *thold = (float *)malloc(sizeof(float));
  float *vars = (float *)malloc(sizeof(float)*nprocess);

  memset(hOutputData, 0, size);
  memset(hist, 0, histSize);
  memset(wB, 0, histSize);
  memset(sumB, 0, histSize);
  memset(thold, 0, sizeof(float));

  float thresholds[nprocess];
  
  //start
  start = MPI_Wtime();
  
  double tempTime = 0;
  int numPix = width*height;
  int numPerNode = numPix / nprocess;
  int from = rank * numPerNode;
  int till = 0;

  if(rank == (nprocess-1))
    till = numPix;
  else
    till = from + numPerNode - 1;

  calcHist(hData, hist, sumB, wB, width, height, from, till, rank);

  overheadS = MPI_Wtime();
  MPI_Bcast(sumB, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(wB, 256, MPI_FLOAT, 0, MPI_COMM_WORLD);
  overheadE = MPI_Wtime();
  overhead += overheadE - overheadS;
  tempTime += overheadE - overheadS;
  
  numPerNode = 256/nprocess;
  from  = rank * numPerNode;
  till  = from + numPerNode - 1; 

  otsu(from, till, hist, sumB, wB, thold, vars, width, height);

  overheadS = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(thold, 1, MPI_FLOAT, thresholds, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  overheadE = MPI_Wtime();
  overhead += overheadE - overheadS;
  tempTime += overheadE - overheadS;
  
  int k = 0;
  if(rank == 0)
    k = findMax(vars, nprocess);

  thold[0] = thresholds[k];
  overheadS = MPI_Wtime();
  MPI_Bcast(thold, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  overheadE = MPI_Wtime();
  overhead += overheadE - overheadS;
  tempTime += overheadE - overheadS;

  if(rank == 0)
    threshold(hData, hOutputData, thold, width, height);

  end = MPI_Wtime();  

  if(rank == 0){
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(outputFilename) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
  }

  printf("Rank : %d, Processing Time : %f, Overhead : %f\n", rank, (end-start-tempTime)*1000, overhead*1000);

  free(hData);
  free(hist);
  free(hOutputData);

  MPI_Finalize();

  return 0;
}
