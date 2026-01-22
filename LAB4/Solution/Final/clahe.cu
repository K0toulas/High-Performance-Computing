#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clahe.h"

#define PAD(i)  ( (i) + ((i) >> 5) )

// Helper: Read PGM
__host__ PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;
    cudaError_t err_runtime;

    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip P5*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d",&v_max);
    fgetc(in_file); // Skip the single whitespace/newline after max_val

    err_runtime = cudaMallocHost((void**)&result.img, result.w * result.h * sizeof(unsigned char));
    if (err_runtime != cudaSuccess) {
        cudaDeviceReset();
        exit(1); 
    }

    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

// Helper: Write PGM
__host__ void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

// Helper: Free PGM Memory
__host__ void free_pgm(PGM_IMG img) {
    if(img.img) free(img.img);
}

__global__ void lut_init(int *d_all_luts, const unsigned char *__restrict__ d_img_in, int height, \
                         int width, int grid_w, int grid_h) {
    int tx, ty, x_start, y_start, actual_tile_w, actual_tile_h;
    int *current_lut_ptr;
    int x, y, i, avg_inc, val;
    int cdf = 0, total_pixels;
    int tid;
    int stride, idx, swap_val, excess_local;
    
    __shared__ int priv_hist[THREAD_TEAMS][256];
    __shared__ int hist[256];
    __shared__ int excess;
    
    tx = blockIdx.x;
    ty = blockIdx.y;

    if (tx >= grid_w || ty >= grid_h)
        return;

    // Precompute all Tile LUTs
    x_start = tx * TILE_SIZE;
    y_start = ty * TILE_SIZE;
            
    // Handle boundary tiles that might be smaller than TILE_SIZE
    // Change ternary to min to avoid branches - divergence
    actual_tile_w = min(TILE_SIZE, width  - x_start);
    actual_tile_h = min(TILE_SIZE, height - y_start);

    // Pointer to the specific 256-entry LUT for this tile
    current_lut_ptr = &d_all_luts[(ty * grid_w + tx) * 256];
 
    // 0...BLOCK_DIM*BLOCK_DIM-1
    tid = threadIdx.y * blockDim.x + threadIdx.x;
    total_pixels = actual_tile_w * actual_tile_h;

    if(tid == 0) 
        excess = 0;
    // initialize histogram, only first 256 threads
    // if threads per block more than 256 its ok
    for (i=0; i < THREAD_TEAMS; i++) 
        priv_hist[i][tid]=0;
       
     hist[tid] = 0;
     __syncthreads();

    int hist_idx = tid / THREADS_PER_TEAM;
    // Build Histogram
    for (i=tid; i < actual_tile_h*actual_tile_w; i+=TOTAL_THREADS) {
        x = x_start + (i % actual_tile_w);
        y = y_start + (i / actual_tile_w);
        // Boundary check mostly for the right/bottom edge tiles
        if (x < width && y < height) {
            atomicAdd(&(priv_hist[hist_idx][d_img_in[y * width + x]]), 1);
        }
    }
    __syncthreads();

    for (i=0; i < THREAD_TEAMS; i++) {
        hist[tid] += priv_hist[i][tid];
    }
    __syncthreads();

    // Clip Histogram
    // remove if statement 
    excess_local = max(0, hist[tid] - CLIP_LIMIT);
    atomicAdd(&excess, excess_local);
    hist[tid] = min(hist[tid], CLIP_LIMIT);

    __syncthreads();

    // Redistribute Excess (simplisticly)
    avg_inc = excess / 256;
    hist[tid] += avg_inc;
    __syncthreads();

    __shared__ int temp[256 + 8]; 

    // Load with padding
    temp[PAD(tid)] = hist[tid];
    __syncthreads();

    // reduction phase
    for (stride = 1; stride < 256; stride <<= 1) {
        idx = (tid + 1) * stride * 2 - 1;
        if (idx < 256)
            temp[PAD(idx)] += temp[PAD(idx - stride)];
        __syncthreads();
    }

    // Set last element to zero for exclusive scan
    if (tid == 255)
        temp[PAD(255)] = 0;
    __syncthreads();

    // reverse phase
    for (stride = 128; stride > 0; stride >>= 1) {
        int idx = (tid + 1) * stride * 2 - 1;
        if (idx < 256) {
            swap_val = temp[PAD(idx - stride)];
            temp[PAD(idx - stride)] = temp[PAD(idx)];
            temp[PAD(idx)] += swap_val;
        }
        __syncthreads();
    }

    // Inclusive conversion
    cdf = temp[PAD(tid)] + hist[tid];
    __syncthreads();

    // LUT write
    val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
    //Remove if 
    current_lut_ptr[tid] = min(val, 255);
}  

// Core CLAHE
__global__ void bilinear_interp(const unsigned char *__restrict__ d_img_in, unsigned char *__restrict__ d_img_out, const int *__restrict__ all_luts, int height, int width, int grid_w , int grid_h){
    int x, y, x1, x2, y1, y2, tl, tr, bl, br, val;
    float tx_f, ty_f, x_weight, y_weight, top, bot, final_val;

    // Calculate grid dimensions
    //grid_w = (width + TILE_SIZE - 1) / TILE_SIZE;
    //grid_h = (height + TILE_SIZE - 1) / TILE_SIZE;
    
    x = threadIdx.x + blockIdx.x * blockDim.x;
    y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;
    
    // Render pixels using Bilinear Interpolation
            
    // Find relative position in the grid
    // (y / TILE_SIZE) gives the tile index, but we want the center approach
    // So we offset by 0.5 to align interpolation with tile centers

    //Precompute division
    const float inv_tile = 1.0f / TILE_SIZE;
    ty_f = y * inv_tile - 0.5f;
    tx_f = x * inv_tile - 0.5f;
            
    y1 = (int)floor(ty_f);
    x1 = (int)floor(tx_f);
    y2 = y1 + 1;
    x2 = x1 + 1;

    // Weights for interpolation
    y_weight = ty_f - y1;
    x_weight = tx_f - x1;

    // Clamp tile indices to boundaries 
    // If a pixel is near the edge, it might not have 4 neighbors
    y1 = max(0, min(y1, grid_h - 1));
    y2 = max(0, min(y2, grid_h - 1));
    x1 = max(0, min(x1, grid_w - 1));
    x2 = max(0, min(x2, grid_w - 1));

    //Precompute indexes
    int img_idx = y * width + x;

    int lut_row1 = y1 * grid_w * 256;
    int lut_row2 = y2 * grid_w * 256;
    int lut_x1   = x1 * 256;
    int lut_x2   = x2 * 256;

    // Original pixel intensity
    val =d_img_in[img_idx];
            
    // Fetch mapped values from the 4 nearest tile LUTs
    tl = all_luts[lut_row1 + lut_x1 + val];
    tr = all_luts[lut_row1 + lut_x2 + val];
    bl = all_luts[lut_row2 + lut_x1 + val];
    br = all_luts[lut_row2 + lut_x2 + val];

    // Bilinear interpolation
    top = tl * (1.0f - x_weight) + tr * x_weight;
    bot = bl * (1.0f - x_weight) + br * x_weight;
    final_val = top * (1.0f - y_weight) + bot * y_weight;

    d_img_out[img_idx] = (unsigned char)(final_val + 0.5f);
}