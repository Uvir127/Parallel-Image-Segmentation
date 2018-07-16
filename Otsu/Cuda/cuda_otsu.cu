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
#define num_bits 256

// Texture memory declaration
texture<float, 2, cudaReadModeElementType> imgTex;

//Constant memory variables declaration
__constant__ float constHist[num_bits];

const char *imageFilename = "lena_bw_double.pgm";

//Calculates histogram of image
__global__ void histCalc(float *dHist, int width, int height){
  unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = blockIdx.y * blockDim.x + threadIdx.y;

  if(y < width && x < height){
    int intensity = tex2D(imgTex, x, y)*255;
    atomicAdd(&dHist[intensity], 1);
  }
}

//Scan method used to do partial sum of variables
__global__ void scan(float *oData, const int n, bool isSumB){
  extern __shared__ float sData[];
  int tid = threadIdx.x;
  if(isSumB)
    sData[tid] = tid*constHist[tid];
  else
    sData[tid] = constHist[tid];
  __syncthreads();
  for (int i = 1; i < n; i <<= 1){
    if (tid >= i)
      sData[tid] += sData[tid-i];
    __syncthreads();
  }
  oData[tid] = sData[tid];
}

//Find max value in array by sorting array
__global__ void findMax(const float *iData, float *t){
  extern __shared__ float sData[];
  __shared__ int thresholdIdx[256];

  int tid = threadIdx.x;
  sData[tid] = iData[tid];
  thresholdIdx[tid] = tid;
  __syncthreads();

  for(int s = blockDim.x/2; s >= 1; s>>=1){
    if(tid < s){
      if(sData[tid] < sData[tid + s]){
        sData[tid] = sData[tid + s];
        thresholdIdx[tid] = thresholdIdx[tid + s];
      }
    }
    __syncthreads();
  }

  if(tid == 0)
    *t = thresholdIdx[0];
}

//Otsu thresholding - Finds Highest variance between classes at each threshold
__global__ void otsu(float* wB, float* varB, float* sumB, int total){
  float sum = sumB[255];
  int t = threadIdx.x;

  float wF = 0;

  wF = total - wB[t];             // Weight Foreground
  if (wB[t] != 0 || wF != 0){
    float mF = (sum - sumB[t]) / wF;    // Mean Foreground
    float mB = sumB[t] / wB[t];            // Mean Background

    varB[t] = wB[t] * wF * (mB - mF) * (mB - mF);
    if(isnan(varB[t]))
      varB[t] = 0;
  }
}

//Applies threshold to the image
__global__ void threshold(float* oData, float *t, int height){
  unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int x = blockIdx.y * blockDim.x + threadIdx.y;
  int i = y*height + x;

  float val = tex2D(imgTex, x, y)*255;
  if(val > *t)
    oData[i] = 1;
}

int main(int argc, char *argv[]){
  //Start overhead timer 
  StopWatchInterface *overhead = NULL;
  sdkCreateTimer(&overhead);
  sdkStartTimer(&overhead);

  //Checks if image exists
  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  float *hData = NULL;
  float *dOutputData = NULL;
  float *dHist = NULL;
  float *dSumB = NULL;
  float *wB = NULL;
  float *varB = NULL;
  float *t = NULL;
  char outputFilename[1024];

  sdkLoadPGM(imagePath, &hData, &width, &height);
  int total = width*height;
  size_t size = (width * height) * sizeof(float);
  size_t histSize = num_bits * sizeof(float);

  float *hOutputData = (float *)malloc(size);
  float *hHist = (float *)malloc(histSize);

  cudaMalloc((void **) &dOutputData, size);
  cudaMalloc((void **) &dHist, size);
  cudaMalloc((void **) &dSumB, histSize);
  cudaMalloc((void **) &wB, histSize);
  cudaMalloc((void **) &varB, histSize);
  cudaMalloc((void **) &t, sizeof(float));
  cudaMemset(dHist, 0, histSize);

  // Allocate array and copy image data
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);
  cudaMemcpyToArray(cuArray, 0, 0, hData, size, cudaMemcpyHostToDevice);
  imgTex.filterMode = cudaFilterModePoint;
  cudaBindTextureToArray(imgTex, cuArray, channelDesc);

  StopWatchInterface *timer = NULL;

  dim3 dimBlock(block_size, block_size, 1);
  dim3 dimGrid(ceil((float)width / dimBlock.x), ceil((float)height / dimBlock.y), 1);

  //Start Processing Timer
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  //Calls method that calculates Histogram of Image
  histCalc<<<dimGrid, dimBlock>>>(dHist, width, height);
  cudaDeviceSynchronize();
  cudaMemcpy(hHist, dHist, histSize, cudaMemcpyDeviceToHost);

  //Copies Histogram to constant memory
  cudaMemcpyToSymbol(constHist, hHist, histSize, 0, cudaMemcpyHostToDevice);

  //Does Scan on histogram to give weight and sum background values
  scan<<<1, num_bits, histSize>>>(wB, num_bits, false);
  cudaDeviceSynchronize();
  scan<<<1, num_bits, histSize>>>(dSumB, num_bits, true);
  cudaDeviceSynchronize();

  //Calls otsu kernel to find best threshold
  otsu<<<1, num_bits>>>(wB, varB, dSumB, total);
  cudaDeviceSynchronize();

  //Finds max variance from all the variances calculated
  findMax<<<1, num_bits, histSize>>>(varB, t);
  cudaDeviceSynchronize();

  //Applies threshold to the image
  threshold<<<dimGrid, dimBlock>>>(dOutputData, t, height);
  cudaDeviceSynchronize();

  cudaMemcpy(hOutputData, dOutputData, size, cudaMemcpyDeviceToHost);

  sdkStopTimer(&timer);
  printf("%f\n", sdkGetTimerValue(&timer));

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(outputFilename) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, hOutputData, width, height);

  free(hData);
  free(hHist);
  free(hOutputData);

  cudaFree(dOutputData);
  cudaFree(dHist);
  cudaFree(dSumB);
  cudaFree(wB);
  cudaFree(varB);
  cudaFree(t);
  cudaUnbindTexture(imgTex);

  sdkStopTimer(&overhead);
  printf("%f\n", sdkGetTimerValue(&overhead)-sdkGetTimerValue(&timer));
  sdkDeleteTimer(&overhead);
  return 0;
}
