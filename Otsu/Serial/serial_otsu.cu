// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

const char *imageFilename = "lena_bw_triple.pgm";

void otsu(float* iData, float* oData, float* hist,  int width, int height){
  int index = 0;

  // Total number of pixels
  int total = width*height;

  // Calculate histogram
  while (index < total) {
    int h = iData[index]*255;
    hist[h]++;
    index++;
  }

  float sum = 0;
  for (int t=0 ; t<=255 ; t++){
    sum += t * hist[t];
  }

  float sumB = 0;
  float wB = 0;
  float wF = 0;

  float maxVar = 0;
  float threshold = 0;

  for (int t=0 ; t<256 ; t++) {
    wB += hist[t];               // Weight Background
    wF = total - wB;             // Weight Foreground

    if (wB != 0 || wF != 0){
      sumB += t * hist[t];

      float mF = (sum - sumB) / wF;    // Mean Foreground
      float mB = sumB / wB;            // Mean Background

      // Calculate Between Class Variance
      float varB = wB * wF * (mB - mF) * (mB - mF);
      if(isnan(varB))
        varB = 0;
      // Check if new maximum found
      if (varB > maxVar) {
        maxVar = varB;
        threshold = t;
      }
    }
  }

  for (size_t i = 0; i < total; i++) {
    float val = iData[i]*255;
    if(val > threshold)
      oData[i] = 1;
  }
}

int main(int argc, char *argv[]){
  //Starts overhead timer  
  StopWatchInterface *overhead = NULL;
  sdkCreateTimer(&overhead);
  sdkStartTimer(&overhead);

  //Checks if file exists
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

  memset(hOutputData, 0, size);
  memset(hist, 0, histSize);

  StopWatchInterface *timer = NULL;

  //Starts processing timer
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  otsu(hData, hOutputData, hist, width, height);

  sdkStopTimer(&timer);
  printf("%f\n", sdkGetTimerValue(&timer));

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(outputFilename) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, hOutputData, width, height);

  free(hData);
  free(hist);
  free(hOutputData);

  sdkStopTimer(&overhead);
  printf("%f\n", sdkGetTimerValue(&overhead)-sdkGetTimerValue(&timer));
  sdkDeleteTimer(&overhead);
  return 0;
}
