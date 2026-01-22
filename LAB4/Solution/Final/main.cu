#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "clahe.h"

#define IS_NULL(ptr) (((ptr) == (NULL)) ? 1 : 0)
#define IS_CUDA_MALLOC_OK(err) (((err) == (cudaSuccess)) ? (1) : (0))
#define FREE_MEM(ptr) (((ptr) != (NULL)) ? (free(ptr)) : ((void)(ptr))) // Free CPU mem
// Check for run time error and free allocated memory if you detect failure
#define CHECK_CUDA_FAIL(err)\
({\
  if (err != cudaSuccess) {\
    printf("CUDA error: %s (%d) in %s, line %i\n", cudaGetErrorString(err), err, __FILE__, __LINE__);\
    /* free host mem */\
    FREE_MEM(h_img_in.img);\
    FREE_MEM(h_img_in.img);\
    /* free device mem */\
    cudaFree(d_img_in.img);\
    cudaFree(d_img_out.img);\
    cudaFree(d_all_luts);\
    /* Reset device just in case */\
    cudaDeviceReset();\
    return 1;\
    }\
})

double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]){
    PGM_IMG h_img_in, h_img_out;
    PGM_IMG d_img_in, d_img_out;

    int height, width;
    int grid_w, grid_h;
    int blocks_y, blocks_x;
    cudaError_t error_input, error_output, error_luts;
    cudaError_t error_runtime, error_out_h;
    int *d_all_luts;
    double start, end, elapsed;

    if (argc != 3) {
        printf("Usage: %s <input.pgm> <output.pgm>\n", argv[0]);
        return 1;
    }

    printf("Loading image...\n");
    h_img_in = read_pgm(argv[1]);
    height = h_img_in.h;
    width = h_img_in.w;
    h_img_out.w = width;
    h_img_out.h = height;

    printf("Height: %d, width: %d\n", height, width);

    // calculate grid dimensions
    grid_w = (width + TILE_SIZE - 1) / TILE_SIZE;
    grid_h = (height + TILE_SIZE - 1) / TILE_SIZE;
    printf("grid_w: %d, grid_h: %d\n", grid_w, grid_h);
    
    printf("Max Block dims (%d, %d)\n", BLOCK_DIM, BLOCK_DIM);
    printf("Lut_init grid dims (%d, %d)\n", grid_w, grid_h);
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM); 
    dim3 grid_dim_lut(grid_w, grid_h);

    // for bilinear interpolation kernel, one pixel per thread 
    blocks_x = (width + BLOCK_DIM - 1) / BLOCK_DIM;
    blocks_y = (height + BLOCK_DIM - 1) / BLOCK_DIM;
    printf("bilinear_interp grid dims (%d, %d)\n", blocks_x, blocks_y);
    dim3 grid_dim_bilinear(blocks_x, blocks_y);

    start = get_time_sec();

    // allocate memory on device 
    error_input = cudaMalloc((void **)&d_img_in.img, height*width*sizeof(unsigned char));
    error_output = cudaMalloc((void **)&d_img_out.img, height*width*sizeof(unsigned char));
    error_out_h = cudaMallocHost((void**)&h_img_out.img, height*width*sizeof(unsigned char));
    error_luts = cudaMalloc((void **)&d_all_luts, grid_h * grid_w * 256 * sizeof(int));
    if(!IS_CUDA_MALLOC_OK(error_input) || !IS_CUDA_MALLOC_OK(error_output) || \
        !IS_CUDA_MALLOC_OK(error_luts) || !IS_CUDA_MALLOC_OK(error_out_h)) {
        printf("CUDA: Memory allocation Failed!\n");
        FREE_MEM(h_img_out.img);
        FREE_MEM(h_img_in.img);
        IS_CUDA_MALLOC_OK(error_input) ? cudaFree(d_img_in.img) : 0;
        IS_CUDA_MALLOC_OK(error_output) ? cudaFree(d_img_out.img) : 0;
        IS_CUDA_MALLOC_OK(error_luts) ? cudaFree(d_all_luts) : 0;
        return 1;
    }

    cudaStream_t s1;
    error_runtime = cudaStreamCreate(&s1);
    CHECK_CUDA_FAIL(error_runtime);

    error_runtime = cudaMemcpyAsync(d_img_in.img, h_img_in.img, height*width*sizeof(unsigned char), \
                                    cudaMemcpyHostToDevice, s1);
    CHECK_CUDA_FAIL(error_runtime);

    // launch lut_init kernel
    lut_init<<<grid_dim_lut, block_dim, 0, s1>>>(d_all_luts, d_img_in.img, height, width, grid_w, grid_h);
    error_runtime = cudaGetLastError();
    CHECK_CUDA_FAIL(error_runtime);

    // launch bilinear_interp kernel
    bilinear_interp<<<grid_dim_bilinear, block_dim, 0, s1>>>(d_img_in.img, d_img_out.img, d_all_luts, height, width, grid_w, grid_h);
    error_runtime = cudaGetLastError();
    CHECK_CUDA_FAIL(error_runtime);    

    error_runtime = cudaMemcpyAsync(h_img_out.img, d_img_out.img, height*width*sizeof(unsigned char), \
                                    cudaMemcpyDeviceToHost, s1);
    CHECK_CUDA_FAIL(error_runtime);

    error_runtime = cudaStreamSynchronize(s1);
    CHECK_CUDA_FAIL(error_runtime);
    
    end = get_time_sec();
    elapsed = end - start;
    
    printf("Processing time: %.6f seconds\n", elapsed);
    printf("Throughput: %.2f MPixels/s\n", (h_img_in.w * h_img_in.h) / (elapsed * 1e6));
    
    cudaStreamDestroy(s1);

    write_pgm(h_img_out, argv[2]);
    printf("Result saved to %s\n", argv[2]);

    cudaFreeHost(h_img_in.img);
    cudaFreeHost(h_img_out.img);

    cudaFree(d_img_in.img);
    cudaFree(d_img_out.img);
    cudaFree(d_all_luts);

    cudaDeviceReset(); 
    return 0;
}

