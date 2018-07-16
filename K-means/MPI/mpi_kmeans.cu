// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "time.h"
#include <mpi.h>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define k 16

const char *imageFilename = "test.ppm";

void kmeans(unsigned char *g_idata,unsigned char *g_odata, unsigned int width, unsigned int height,float * clus,int start,int finish,float* cTemp,float* count){
// same as above
  float newsum=0.0;
  float r=0.0;
  float g=0.0;
  float b=0.0;
  float sum=100000.0;
  int cVal=0;
  unsigned int size = k;
  for(int i=start;i<finish;i++){
    for(int j=0;j<width;j++){

      newsum=0.0;
      r=0.0;
      g=0.0;
      b=0.0;
      sum=10000000.0;
      cVal=0;

      for(int a=0;a<size;a++){

        newsum=(abs((clus[a*3])-(g_idata[i*width*4+j*4]))+abs(clus[a*3+1]-(g_idata[i*width*4+j*4+1]))+abs(clus[a*3+2]-(g_idata[i*width*4+j*4+2])));
        if(sum>newsum){
          cVal=a;
          sum=newsum;
          r=(g_idata[i*width*4+j*4]);
          g=(g_idata[i*width*4+j*4+1]);
          b=(g_idata[i*width*4+j*4+2]);
          g_odata[i*width*4+j*4]=cVal;
          g_odata[i*width*4+j*4+1]=cVal;
          g_odata[i*width*4+j*4+2]=cVal;
        }
      }


      count[cVal]+=1.0;
      cTemp[cVal*3]+=r;
      cTemp[cVal*3+1]+=g;
      cTemp[cVal*3+2]+=b;
 }
}

}

void serial(unsigned char *g_idata,unsigned char *g_odata, unsigned int width, unsigned int height,float * clus){


  int stop=0;
  float*  cTemp = (float *)malloc(k*3* sizeof(float));// pass through , dont declare locally
  float*  count = (float *)malloc(k* sizeof(float));
  float*  cTempS = (float *)malloc(k*3* sizeof(float));// pass through , dont declare locally
  float*  countS = (float *)malloc(k* sizeof(float));
  unsigned int size = k;
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for(int i=0;i<size;i++){
    cTempS[i*3]=0;
    cTempS[i*3+1]=0;
    cTempS[i*3+2]=0;
    countS[i]=0;
  }
  int num=0;
  MPI_Comm_size(MPI_COMM_WORLD, &num);
  while(stop==0){
    for(int i=0;i<size;i++){
      cTemp[i*3]=0;
      cTemp[i*3+1]=0;
      cTemp[i*3+2]=0;
      count[i]=0;
    }
      if(rank==num-1){
        kmeans(g_idata, g_odata, width, height, clus,floor(height/(num-1))*rank,height,cTemp,count);
      }else{
        kmeans(g_idata, g_odata, width, height, clus,floor(height/(num-1))*rank,floor(height/(num-1))*(rank+1),cTemp,count);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Allreduce(cTemp, cTempS, k*3 ,  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(count, countS, k ,  MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      if(rank==0){
        float error=0.0;
        for(int a=0;a<size;a++){
          if(countS[a]!=0.0){
            error=error+abs(clus[a*3]-(cTempS[a*3]/countS[a]))+abs(clus[a*3+1]-(cTempS[a*3+1]/countS[a]))+abs(clus[a*3+2]-(cTempS[a*3+2]/countS[a]));
          }else{
            printf("%d\n",size);
          }
        }
        if(error/(float)size<10){
          stop=1;
        //  MPI_Bcast(&stop,1,MPI_INT,0,MPI_COMM_WORLD);
          printf("%d\n",size );
        }
        for(int s=0;s<size;s++){
          if(countS[s]!=0){
          clus[s*3]=cTempS[s*3]/countS[s];
          clus[s*3+1]=cTempS[s*3+1]/countS[s];
          clus[s*3+2]=cTempS[s*3+2]/countS[s];
         }


        }
      }
      MPI_Bcast(&stop,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(clus,k*3,MPI_FLOAT,0,MPI_COMM_WORLD);
      }
      unsigned int size1 = width * height*4;
      unsigned char * g_odataf = (unsigned char *)malloc(size1*sizeof(unsigned char));
      MPI_Allreduce(g_odata,g_odataf,size1,MPI_UNSIGNED_CHAR,MPI_SUM,MPI_COMM_WORLD);



  //}
if(rank==0){
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      g_odata[i*width*4+j*4]=clus[g_odataf[i*width*4+j*4]*3];
      g_odata[i*width*4+j*4+1]=clus[g_odataf[i*width*4+j*4+1]*3+1];
      g_odata[i*width*4+j*4+2]=clus[g_odataf[i*width*4+j*4+2]*3+2];
    }
  }
}

}
int main(int argc, char **argv){
  MPI_Init(&argc, &argv);
  StopWatchInterface *timer1 = NULL;
  double start,finish,proS,proF;
  start=MPI_Wtime();
  char *imagePath = sdkFindFilePath(imageFilename, "");
  if (imagePath == NULL)
  {
      printf("Unable to source image file: %s\n", imageFilename);
      exit(EXIT_FAILURE);
  }

  unsigned int width, height;
  time_t t;
  srand(time(&t));
  unsigned char *hData = NULL;
  //float *hOutputData = NULL;
  char outputFilename[1024];

  sdkLoadPPM4ub(imagePath, &hData, &width, &height);
  int csize=k;
  unsigned int size = width * height * sizeof(unsigned char)*4;
  unsigned char * hOutputData = (unsigned char *)malloc(size);
  float* clusH = (float *)malloc(csize*3* sizeof(float));
  float* clusT = (float *)malloc(csize*3* sizeof(float));
  float* counter = (float *)malloc(csize* sizeof(float));
  // Bind the array to the texture
  int rank=0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(rank==0){
  for(int i=0;i<csize;i++){
    int randomnumber;
    randomnumber = rand() % width*height;
    clusH[i*3]=hData[randomnumber*4];
    clusH[i*3+1]=hData[randomnumber*4+1];
    clusH[i*3+2]=hData[randomnumber*4+2];

  }
}
  MPI_Bcast(clusH,k*3,MPI_FLOAT,0,MPI_COMM_WORLD);
  proS=MPI_Wtime();
  serial(hData, hOutputData, width, height, clusH);
  proF=MPI_Wtime()-proS;

  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out1.ppm");
  if(rank==0){
  sdkSavePPM4ub(outputFilename, hOutputData, width, height);
   }
  finish=MPI_Wtime()-start;
  printf("Processing time: %f (ms)\n", proF*1000);
  printf("Overhead time: %f (ms)\n",(finish-proF)*1000);
  sdkDeleteTimer(&timer1);
  MPI_Finalize();
  return 0;
}
