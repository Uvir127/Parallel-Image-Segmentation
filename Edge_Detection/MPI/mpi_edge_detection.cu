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
#include <helper_functions.h> // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h> // helper functions for CUDA error check

const char *imageFilename = "lena_bw_double.pgm";

double overhead, overheadS, overheadE = 0;

void convolution(float *iData, float *oData, int width, int height, int padding, int filterSize, float *filter, int from, int till)
{
  float sum = 0;
  int index = 0;
  int size = width * height;
  float output[size];
  memset(output, 0, size);

  for (int r = padding; r < height - padding; r++)
    for (int c = from; c <= till; c++){
      sum = 0;
      index = c * width + r;
      for (int x = 0; x < filterSize; x++)
        for (int y = 0; y < filterSize; y++)
          sum += filter[y + x * filterSize] * iData[(c + (x - padding)) * width + (r + (y - padding))];
      if (sum < 0)
        output[index] = 0;
      else if (sum > 1)
        output[index] = 1;
      else
        output[index] = sum;
    }

  overheadS = MPI_Wtime();
  MPI_Reduce(&output, oData, size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  overheadE = MPI_Wtime();
  overhead += overheadE - overheadS;
}

//This method pads and unpads the image
void padData(float *iData, float *oData, int paddedWidth, size_t n, const int padding, bool pad)
{
  int paddedEnds = (paddedWidth * padding) + padding;
  int j = 0;
  for (size_t i = paddedEnds; i < n - paddedEnds; i++){
    int mod = i % paddedWidth;
    if (mod >= padding && mod < paddedWidth - padding){
      if (pad)
        oData[i] = iData[j];
      else
        oData[j] = iData[i];
      j++;
    }
  }
}

int main(int argc, char *argv[])
{
  double start, end = 0;
  int nprocess, rank = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocess);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  const int filterSize = 3;
  const int padding = (filterSize - 1) / 2;

  float hfilter[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float *hData = NULL;
  char outputFilename[1024];

  sdkLoadPGM(imagePath, &hData, &width, &height);

  size_t paddedWidth = width + (2 * padding);
  size_t paddedHeight = height + (2 * padding);
  size_t paddedElements = paddedWidth * paddedHeight;
  unsigned int paddedSize = paddedElements * sizeof(float);

  float *hOutputData = (float *)malloc(paddedSize);
  float *hpData = (float *)malloc(paddedSize);

  memset(hpData, 0, paddedSize);
  memset(hOutputData, 0, paddedSize);
  padData(hData, hpData, paddedWidth, paddedElements, padding, true);

  start = MPI_Wtime();

  int numPerNode = floor(width / nprocess);
  int from = rank * numPerNode;
  int till = 0;

  if (rank == 0){
    from += padding;
    till = from + numPerNode - 2;
  }
  else if(rank == (nprocess-1))
    till = width-padding;
  else
    till = from + numPerNode - 1;

  convolution(hpData, hOutputData, paddedWidth, paddedHeight, padding, filterSize, hfilter, from, till);

  end = MPI_Wtime();

  if (rank == 0){
    padData(hOutputData, hData, paddedWidth, paddedElements, padding, false);
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(outputFilename) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hData, width, height);
  }

  printf("Rank : %d, Processing Time : %f, Overhead : %f\n", rank, (end - start) * 1000, overhead * 1000);

  free(hData);
  free(hpData);
  free(hOutputData);

  MPI_Finalize();

  return 0;
}
