/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gputimer.h"

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  	0.00005 
#define MAX_BLOCK_LENGTH 32 // 1024 threads per block (max x, y, coordinates for a block)

// For run time error checking
#define IS_NULL(ptr) (((ptr) == (NULL)) ? 1 : 0)
#define FREE_MEM(ptr) (((ptr) != (NULL)) ? (free(ptr)) : ((void)(ptr))) // Free CPU mem
#define IS_CUDA_MALLOC_OK(err) (((err) == (cudaSuccess)) ? (1) : (0))
// Check for run time error, deallocate memory and return if failure is detected
#define CHECK_CUDA_FAIL(err) \  
({ \
  if (err != cudaSuccess) { \
    printf("CUDA error: %s (%d) in %s, line %i\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
    /* free host mem */ \
    FREE_MEM(h_OutputCPU); \
    FREE_MEM(h_Buffer); \
    FREE_MEM(h_Input); \
    FREE_MEM(h_Filter); \
    FREE_MEM(h_OutputGPU); \
    /* free device mem */ \
    cudaFree(d_Filter); \
    cudaFree(d_Input); \
    cudaFree(d_Buffer); \
    cudaFree(d_OutputGPU); \
    /* Reset device just in case */ \
    cudaDeviceReset(); \
    return 1; \
  } \
})

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}

// Row-wise Convolution for GPU 
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                                  int imageW, int imageH, int filterR) {

  int x, y, k;
  x = threadIdx.x + blockIdx.x * blockDim.x;
  y = threadIdx.y + blockIdx.y * blockDim.y;

  float sum = 0;
  for (k = -filterR; k <= filterR; k++) {
    int d = x + k;

    if (d >= 0 && d < imageW) {
      sum += d_Src[y * imageW + d] * d_Filter[filterR - k];
    }     

    d_Dst[y * imageW + x] = sum;
  } 
}

// Column wise convolution for GPU
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
    			                          int imageW, int imageH, int filterR) {
  int x, y, k;
  x = threadIdx.x + blockIdx.x * blockDim.x;
  y = threadIdx.y + blockIdx.y * blockDim.y;

  float sum = 0;
  for (k = -filterR; k <= filterR; k++) {
    int d = y + k;

    if (d >= 0 && d < imageH) {
      sum += d_Src[d * imageW + x] * d_Filter[filterR - k];
    }   
 
    d_Dst[y * imageW + x] = sum;
  }  
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU; // GPU output on CPU memory

    // devive arrays
    float
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

    // for checking cudaMalloc failure 
    cudaError_t 
    err_filter, 
    err_Input, 
    err_Buffer,
    err_OuputGPU;

    int imageW;
    int imageH;
    unsigned int i, j;

    // For cpu time
    struct timespec  tv1, tv2;
    int stop; // variable to escape outer loop
    // For checking cudaMalloc failure
    cudaError_t err_runtime;
    float diff;
    int grid_length, blockW, blockH;


	  printf("Enter filter radius : ");
	  scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    while(1) {
      stop = 0;
      printf("-----------------------------------\n");
      printf("Image Width x Height = %i x %i\n", imageW, imageH);
      printf("Allocating and initializing host arrays...\n");

      // Memory allocation for CPU
      h_Filter = (float *)malloc(FILTER_LENGTH * sizeof(float));
      h_Input = (float *)malloc(imageW * imageH * sizeof(float));
      h_Buffer = (float *)malloc(imageW * imageH * sizeof(float));
      h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
      h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));

      if (IS_NULL(h_Filter) || IS_NULL(h_Input) || IS_NULL(h_Buffer) || IS_NULL(h_OutputCPU) || IS_NULL(h_OutputGPU)) {
        printf("CPU: Memory allocation Failed!\n");
        FREE_MEM(h_Filter);
        FREE_MEM(h_Input);
        FREE_MEM(h_Buffer);
        FREE_MEM(h_OutputCPU);
        FREE_MEM(h_OutputGPU);
        return 1;
      }

      // Memory allocation for GPU
      err_filter = cudaMalloc((void **)&d_Filter, FILTER_LENGTH * sizeof(float));
      err_Input = cudaMalloc((void **)&d_Input, imageW * imageH * sizeof(float));
      err_Buffer = cudaMalloc((void **)&d_Buffer, imageW * imageH * sizeof(float));
      err_OuputGPU = cudaMalloc((void **)&d_OutputGPU, imageW * imageH * sizeof(float));
      if (!IS_CUDA_MALLOC_OK(err_filter) || !IS_CUDA_MALLOC_OK(err_Input) || !IS_CUDA_MALLOC_OK(err_Buffer) \
          || !IS_CUDA_MALLOC_OK(err_OuputGPU)) {
        printf("CUDA: Memory allocation Failed!\n");
        FREE_MEM(h_Filter);
        FREE_MEM(h_Input);
        FREE_MEM(h_Buffer);
        FREE_MEM(h_OutputCPU);
        FREE_MEM(h_OutputGPU);
        IS_CUDA_MALLOC_OK(err_filter) ? cudaFree(d_Filter) : 0;
        IS_CUDA_MALLOC_OK(err_Input) ? cudaFree(d_Input) : 0;
        IS_CUDA_MALLOC_OK(err_Buffer) ? cudaFree(d_Buffer) : 0;
        IS_CUDA_MALLOC_OK(err_OuputGPU) ? cudaFree(d_OutputGPU) : 0;
        return 1;
      }

      // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
      // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
      // to convolution kai arxikopoieitai kai auth tuxaia.

      srand(200);

      for (i = 0; i < FILTER_LENGTH; i++) {
          h_Filter[i] = (float)(rand() % 16);
      }

      for (i = 0; i < imageW * imageH; i++) {
          h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
      }

      // Send input and filter arrays to device
      err_runtime = cudaMemcpy((float *)d_Input, (float *)h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_FAIL(err_runtime);
      err_runtime = cudaMemcpy((float *)d_Filter, (float *)h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_FAIL(err_runtime);

      // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
      printf("CPU computation...\n");
      clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
      convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
      convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
      clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);


      // Time for CPU
      printf ("CPU: Total time = %10g seconds\n",
        (double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
        (double) (tv2.tv_sec - tv1.tv_sec));


      // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
      // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas
      // how many blocks we need on the grid

      blockH = imageH;
      blockW = imageW;

      if (imageH > MAX_BLOCK_LENGTH) {   
        grid_length = imageH / MAX_BLOCK_LENGTH; 
        printf("grid length %d\n", grid_length);
        blockH = MAX_BLOCK_LENGTH;
        blockW = MAX_BLOCK_LENGTH;
      }
      else {
        // we need only one block
        grid_length = 1;
      }

      dim3 grid_dim(grid_length, grid_length);
      dim3 block_dim(blockW, blockH); 
      
      printf("GPU computation...\n");
      convolutionRowGPU<<<grid_dim, block_dim>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
      cudaDeviceSynchronize();
      err_runtime = cudaGetLastError();
      CHECK_CUDA_FAIL(err_runtime);
      
      convolutionColumnGPU<<<grid_dim, block_dim>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
      cudaDeviceSynchronize();
      err_runtime = cudaGetLastError();
      CHECK_CUDA_FAIL(err_runtime);
      
      // send GPU output to CPU
      cudaMemcpy((float *)h_OutputGPU, (float *)d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);
      
      // Check if GPU output is correct
      for (i = 0; i < imageW; i++) {
        for (j = 0; j < imageH; j++) {
          diff = ABS(*(h_OutputCPU+(i*imageW + j)) - *(h_OutputGPU+(i*imageW + j))); 
          if (diff > accuracy) {
            printf("CPU and GPU output differs! Difference: %f\n", diff);
            stop = 1;
            break;
          }
        }
        if (stop) break;
      }
      
      // free all the allocated memory
      free(h_OutputCPU);
      free(h_Buffer);
      free(h_Input);
      free(h_Filter);
      free(h_OutputGPU);

      // free allocated memory on the device
      cudaFree(d_OutputGPU);
      cudaFree(d_Buffer);
      cudaFree(d_Input);
      cudaFree(d_Filter);    
    
      // Do a device reset just in case...
      cudaDeviceReset();

      imageH *= 2;
      imageW = imageH;
      printf("-----------------------------------\n\n");
    }

    return 0;
}
