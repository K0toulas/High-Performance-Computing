#ifndef CLAHE_H
#define CLAHE_H

#include <stdio.h>
#include <stdlib.h>

// Configuration for CLAHE
// 32x32 is a standard tile size for high-res images
#define TILE_SIZE 32    
// Threshold for contrast limiting (clip limit)    
#define CLIP_LIMIT 2
// Dimension of thread block
#define BLOCK_DIM 16
#define TOTAL_THREADS (BLOCK_DIM*BLOCK_DIM)
#define THREAD_TEAMS 4
#define THREADS_PER_TEAM ((BLOCK_DIM*BLOCK_DIM) / THREAD_TEAMS) 

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;

// I/O functions
__host__ PGM_IMG read_pgm(const char * path);
__host__ void write_pgm(PGM_IMG img, const char * path);
__host__ void free_pgm(PGM_IMG img);

// Core Processing
__global__ void lut_init(int *d_all_luts, const unsigned char *__restrict__ d_img_in, \
                        int height, int width, int grid_w, int grid_h);
__global__ void bilinear_interp(const unsigned char *__restrict__ d_img_in, unsigned char *__restrict__ d_img_out, \
                                 const int *__restrict__ all_luts, int height, int width, int grid_w, int grid_h);

#endif