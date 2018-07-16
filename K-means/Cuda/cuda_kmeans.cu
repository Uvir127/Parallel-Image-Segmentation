// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "time.h"

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

texture<float, 2, cudaReadModeElementType> tex;

#define k 256

const char *imageFilename = "test.ppm";

__device__ float  cTemp[k*3];
__device__ float  count[k];
__device__ bool stop=false;

__global__ void step1(unsigned char *g_idata,unsigned char *g_odata, unsigned int width, unsigned int height,float * clus){
  int i   = threadIdx.y + blockDim.y*blockIdx.y;
  int j   = threadIdx.x + blockDim.x*blockIdx.x;
  int cVal = 0;
  int index = i*width*4+j*4;
  float newsum = 0.0;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  float sum = 100000.0;
  cVal = 0;

  unsigned int size = k;
  for(int a = 0; a < size; a++){
    newsum = abs(clus[a*3]-g_idata[index])+abs(clus[a*3+1]-g_idata[index+1])+abs(clus[a*3+2]-g_idata[index+2]);
    if(sum > newsum){
      cVal = a;
      sum = newsum;
      r = (g_idata[index]);
      g = (g_idata[index+1]);
      b = (g_idata[index+2]);
      g_odata[index] = cVal;
      g_odata[index+1]=cVal;
      g_odata[index+2]=cVal;
    }
  }

  atomicAdd(&count[cVal],1.0);
  atomicAdd(&cTemp[cVal],r);
  atomicAdd(&cTemp[cVal+1],g);
  atomicAdd(&cTemp[cVal+2],b);
  __syncthreads();
  }


__global__ void step2(unsigned char *g_idata,unsigned char *g_odata, unsigned int width, unsigned int height,float * clus){
  int i=threadIdx.y + blockDim.y*blockIdx.y;
  int j=threadIdx.x + blockDim.x*blockIdx.x;
  int index = i*width*4+j*4;
  g_odata[index]    = clus[g_odata[index]*3];
  g_odata[index+1]  = clus[g_odata[index+1]*3+1];
  g_odata[index+2]  = clus[g_odata[index+2]*3+2];
}

int main(int argc, char **argv){
  StopWatchInterface *timer1 = NULL;
  sdkCreateTimer(&timer1);
  sdkStartTimer(&timer1);
  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL){
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  time_t t;
  srand(time(&t));
  unsigned char *hData = NULL;
  unsigned char *dData = NULL;
  unsigned char *dOut = NULL;
  float *dClus = NULL;
  char outputFilename[1024];

  sdkLoadPPM4ub(imagePath, &hData, &width, &height);
  int csize=k;
  unsigned int size = width * height * sizeof(unsigned char)*4;
  unsigned char * hOutputData = (unsigned char *)malloc(size);
  float* clusH = (float *)malloc(csize*3* sizeof(float));
  float* clusT = (float *)malloc(csize*3* sizeof(float));
  float* counter = (float *)malloc(csize* sizeof(float));

  cudaMalloc((void **) &dData, size);
  cudaMalloc((void **) &dOut, size);
  cudaMalloc((void **) &dClus, csize*3*sizeof(float));

  for(int i=0;i<csize;i++){
    int randomnumber;
    randomnumber  = rand() % width*height;
    clusH[i*3]    = hData[randomnumber*4];
    clusH[i*3+1]  = hData[randomnumber*4+1];
    clusH[i*3+2]  = hData[randomnumber*4+2];
  }

  cudaMemcpy(dData, hData, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dClus, clusH , csize*3*sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  bool stop = false;
  while(stop == false){
    step1<<<dimGrid,dimBlock>>>(dData, dOut, width, height, dClus);
    cudaDeviceSynchronize();
    float error=0;

    cudaMemcpyFromSymbol(clusT, cTemp, csize*3*sizeof(float), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(counter, count, csize*sizeof(float), 0, cudaMemcpyDeviceToHost);

      for(int a = 0; a < csize; a++){
        if(counter[a] != 0){
          error += abs(clusH[a*3]-clusT[a*3]/counter[a])+abs(clusH[a*3+1]-clusT[a*3+1]/counter[a])+abs(clusH[a*3+2]-clusT[a*3+2]/counter[a]);
        }
      }

      if(error/csize<10){
        stop=true;
      }

      for(int s=0;s<csize;s++){
        if(counter[s]!=0){
          clusH[s*3]=clusT[s*3]/counter[s];
          clusH[s*3+1]=clusT[s*3+1]/counter[s];
          clusH[s*3+2]=clusT[s*3+2]/counter[s];
        }
        clusT[s*3]=0.0;
        clusT[s*3+1]=0.0;
        clusT[s*3+2]=0.0;
        counter[s]=0;
      }

      cudaMemcpyToSymbol(dClus, clusH , csize*3*sizeof(float),0, cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(cTemp,clusT , csize*3*sizeof(float),0, cudaMemcpyHostToDevice);
      cudaMemcpyToSymbol(count,counter , csize*sizeof(float),0, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
  }

  step2<<<dimGrid,dimBlock>>>(dData, dOut, width, height, dClus);
  cudaDeviceSynchronize();

  sdkStopTimer(&timer);

  cudaMemcpy(hOutputData, dOut, size, cudaMemcpyDeviceToHost);

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out1.ppm");
  sdkSavePPM4ub(outputFilename, hOutputData, width, height);
  sdkStopTimer(&timer1);

  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  printf("Overhead time: %f (ms)\n",(sdkGetTimerValue(&timer1)- sdkGetTimerValue(&timer)));
  sdkDeleteTimer(&timer1);
  return 0;
}
