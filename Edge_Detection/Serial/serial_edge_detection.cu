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

const char *imageFilename = "lena_bw_double.pgm";

//Application of filter to image
void convolution(float* iData, float* oData, int width, int height, int padding, int filterSize, float *filter){
  float sum = 0;
  for(int r = padding; r < height-padding; r++)
    for(int c = padding; c < width-padding; c++){
      sum = 0;
      for(int x = 0; x<filterSize; x++)
				for(int y = 0; y<filterSize; y++)
          sum += filter[y + x*filterSize]*iData[(c+(x-padding))*width + (r+(y-padding))];
      if(sum < 0)
        oData[c*width+r] = 0;
      else if (sum > 1)
        oData[c*width+r] = 1;
      else
        oData[c*width+r] = sum;
    }
}

//This method pads and unpads the image
void padData(float* iData, float* oData, int paddedWidth, size_t n, const int padding, bool pad){
  int paddedEnds  = (paddedWidth * padding) + padding;

  int j = 0;
  for (size_t i = paddedEnds; i < n-paddedEnds; i++) {
    int mod = i%paddedWidth;
    if(mod >= padding && mod < paddedWidth-padding){
      if(pad)
        oData[i] = iData[j];
      else
        oData[j] = iData[i];
      j++;
    }
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
  const int filterSize = 3;
  const int padding = (filterSize-1)/2;

  int filLength = pow(filterSize, 2);
  float hfilter[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1 };

  float *hData = NULL;
  char outputFilename[1024];

  sdkLoadPGM(imagePath, &hData, &width, &height);

  //Gets padded data set up
  size_t paddedWidth      = width + (2 * padding);
  size_t paddedHeight     = height + (2 * padding);
  size_t paddedElements   = paddedWidth * paddedHeight;
  unsigned int paddedSize = paddedElements * sizeof(float);

  float *hOutputData = (float *)malloc(paddedSize);
  float *hpData = (float *)malloc(paddedSize);

  memset(hpData, 0, paddedSize);
  padData(hData, hpData, paddedWidth, paddedElements, padding, true);

  StopWatchInterface *timer = NULL;

  //Starts processing timer
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  convolution(hpData, hOutputData, paddedWidth, paddedHeight, padding, filterSize, hfilter);

  sdkStopTimer(&timer);
  printf("%f\n", sdkGetTimerValue(&timer));

  padData(hOutputData, hData, paddedWidth, paddedElements, padding, false);

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(outputFilename) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, hData, width, height);

  free(hData);
  free(hpData);
  free(hOutputData);

  sdkStopTimer(&overhead);
  printf("%f\n", sdkGetTimerValue(&overhead)-sdkGetTimerValue(&timer));
  sdkDeleteTimer(&overhead);
  return 0;
}
