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

#define block_size 16

const char *imageFilename = "lena_bw_double.pgm";

// Texture memory declaration
texture<float, 2, cudaReadModeElementType> imgTex;

//Constant memory variables declaration
__constant__ int constWidth[1];
__constant__ int constHeight[1];
__constant__ int constFilterSize[1];
__constant__ int constPadding[1];
__constant__ float constFilter[9];

//Application of filter to image
__global__ void convolution(float *oData){
  unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int c = blockIdx.y * blockDim.x + threadIdx.y;

  float sum = 0;

  if((r >= constPadding[0] && r < constHeight[0]-constPadding[0]) && (c >= constPadding[0] && c < constHeight[0]-constPadding[0]))
    sum = 0;
    for(int x = 0; x<constFilterSize[0]; x++)
    	for(int y = 0; y<constFilterSize[0]; y++)
        sum += constFilter[y + x*constFilterSize[0]]*tex2D(imgTex, (r+(y-constPadding[0])), c+(x-constPadding[0]));
    if(sum < 0)
      oData[c*constWidth[0]+r] = 0;
    else if (sum > 1)
      oData[c*constWidth[0]+r] = 1;
    else
      oData[c*constWidth[0]+r] = sum;
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

int main(int argc, char **argv){
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
  //initializes needed variables
  int filterSize[1] = {3};
  int padding[1] = {(filterSize[0]-1)/2};

  int filLength = pow(filterSize[0], 2);
  float hfilter[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1 };

  float *hData = 0;
  float *dData = 0;
  float *dOutputData = 0;
  float *filter = 0;
  char outputFilename[1024];

  sdkLoadPGM(imagePath, &hData, &width, &height);

  //Gets padded data set up
  int paddedWidth[1]      = {width + (2 * padding[0])};
  int paddedHeight[1]     = {height + (2 * padding[0])};
  size_t paddedElements   = paddedWidth[0] * paddedHeight[0];
  unsigned int paddedSize = paddedElements * sizeof(float);

  float *hOutputData = (float *)malloc(paddedSize);
  float *hpData      = (float *)malloc(paddedSize);
  cudaMalloc((void **) &dData, paddedSize);
  cudaMalloc((void **) &dOutputData, paddedSize);
  cudaMalloc((void **) &filter, filLength*sizeof(float));

  padData(hData, hpData, paddedWidth[0], paddedElements, padding[0], true);

  cudaMemcpy(dData, hpData, paddedSize, cudaMemcpyHostToDevice);
  cudaMemcpy(filter, hfilter, filLength*sizeof(float), cudaMemcpyHostToDevice);

  //Copies variables to constant memory
  cudaMemcpyToSymbol(constWidth, paddedWidth, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(constHeight, paddedHeight, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(constFilterSize, filterSize, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(constFilter, hfilter, filLength*sizeof(float), 0, cudaMemcpyHostToDevice);

  // Allocate array and copy image data
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, paddedWidth[0], paddedHeight[0]);
  cudaMemcpyToArray(cuArray, 0, 0, hpData, paddedSize, cudaMemcpyHostToDevice);
  imgTex.filterMode = cudaFilterModePoint;
  cudaBindTextureToArray(imgTex, cuArray, channelDesc);

  dim3 dimBlock(block_size, block_size, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  //Starts processing timer
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  convolution<<<dimGrid, dimBlock>>>(dOutputData);
  cudaDeviceSynchronize();

  sdkStopTimer(&timer);
  printf("%f\n", sdkGetTimerValue(&timer));

  cudaMemcpy(hOutputData, dOutputData, paddedSize, cudaMemcpyDeviceToHost);

  padData(hOutputData, hData, paddedWidth[0], paddedElements, padding[0], false);

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(outputFilename) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, hData, width, height);

  free(hData);
  free(hOutputData);
  cudaFree(dData);
  cudaFree(dOutputData);
  cudaFree(filter);

  sdkStopTimer(&overhead);
  printf("%f\n", sdkGetTimerValue(&overhead)-sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);
  sdkDeleteTimer(&overhead);

  return 0;
}
