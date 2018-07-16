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

#define k 32

const char *imageFilename = "test.ppm";

void serial(unsigned char *g_idata,unsigned char *g_odata, unsigned int width, unsigned int height,float * clus){
  bool stop     = false;
  float newsum  = 0.0;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  float sum = 100000.0;
  int cVal = 0;
  int index = 0;
  int size = k;
  float cTemp [k*3];
  float count[k];
  while(stop == false){
    for(int i = 0; i < size; i++){
      cTemp[i*3]   = 0;
      cTemp[i*3+1] = 0;
      cTemp[i*3+2] = 0;
      count[i] = 0;
    }

    for(int i = 0; i < height; i++){
      for(int j = 0; j < width; j++){
        newsum = 0.0;
        r = 0.0;
        g = 0.0;
        b = 0.0;
        sum = 10000000.0;
        cVal = 0;
        index = i*width*4+j*4;
        for(int a = 0; a < size; a++){
          newsum=abs(clus[a*3]-g_idata[index])+abs(clus[a*3+1]-g_idata[index+1])+abs(clus[a*3+2]-g_idata[index+2]);
          if(sum > newsum){
            cVal = a;
            sum  = newsum;
            r = (g_idata[index]);
            g = (g_idata[index+1]);
            b = (g_idata[index+2]);
            g_odata[index]=cVal;
            g_odata[index+1]=cVal;
            g_odata[index+2]=cVal;
          }
        }


        count[cVal]++;
        cTemp[cVal*3]   += r;
        cTemp[cVal*3+1] += g;
        cTemp[cVal*3+2] += b;
   }
  }

  float error = 0.0;
  for(int a = 0; a < size; a++){
    if(count[a] != 0.0){
      error += abs(clus[a*3]-(cTemp[a*3]/count[a]))+abs(clus[a*3+1]-(cTemp[a*3+1]/count[a]))+abs(clus[a*3+2]-(cTemp[a*3+2]/count[a]));
    }
  }

  if(error/(float)size<10){
    stop=true;
    printf("%d\n",size );
  }

  for(int s = 0; s < k; s++){
    if(count[s] != 0.0){
    clus[s*3]   = cTemp[s*3]/count[s];
    clus[s*3+1] = cTemp[s*3+1]/count[s];
    clus[s*3+2] = cTemp[s*3+2]/count[s];
  }
  cTemp[s*3]    = 0;
  cTemp[s*3+1]  = 0;
  cTemp[s*3+2]  = 0;
  count[s] = 0;
  }

  }
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      index = i*width*4+j*4;
      g_odata[index]    = clus[g_odata[index]*3];
      g_odata[index+1]  = clus[g_odata[index+1]*3+1];
      g_odata[index+2]  = clus[g_odata[index+2]*3+2];
    }
  }

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
  unsigned char *dOut = NULL;
  char outputFilename[1024];

  sdkLoadPPM4ub(imagePath, &hData, &width, &height);

  int csize = k;
  unsigned int size           = width * height * sizeof(unsigned char)*4;
  unsigned char * hOutputData = (unsigned char *)malloc(size);
  float* clusH    = (float *)malloc(csize*3* sizeof(float));
  float* clusT    = (float *)malloc(csize*3* sizeof(float));
  float* counter  = (float *)malloc(csize* sizeof(float));

  for(int i=0;i<csize;i++){
    int randomnumber;
    randomnumber = rand() % width*height;
    clusH[i*3]=hData[randomnumber*4];
    clusH[i*3+1]=hData[randomnumber*4+1];
    clusH[i*3+2]=hData[randomnumber*4+2];
  }

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  serial(hData, hOutputData, width, height, clusH);
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
