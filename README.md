# High Performance Computing (HPC) Assignments

This repository contains the laboratory assignments for the High Performance Computing (ECE 415) course at the University of Thessaly.

## Repository Structure

Each lab folder contains the original assignment specifications.

* **Solutions:** The source code solutions for each assignment are located in the `Solution` subdirectory within each lab folder.
* **Reports:** For detailed analysis, performance metrics, and implementation notes, please refer to the **Report PDF** included in each solution folder.

---

## Assignment Summaries

### Lab 1: Sequential Code Optimization [C / Sequential]
This assignment focuses on the step by step optimization of a sequential C program that performs edge detection using the Sobel filter. The goal is to apply various techniques such as loop unrolling, loop fusion and function inlining to improve execution time while verifying correctness against a baseline implementation.

### Lab 2: K-Means Parallelization [C / OpenMP]
This lab involves parallelizing a sequential k-means clustering algorithm using OpenMP to utilize multi-core processors effectively. The task requires profiling the code to identify bottlenecks, implementing parallel regions with appropriate scheduling and evaluating performance scaling across different thread counts.

### Lab 3: Image Convolution [C / CUDA]
This assignment introduces GPU programming with CUDA by implementing a 2D convolution filter (separable filter) on an image. It involves managing memory between host and device, handling thread divergence and addressing precision issues when comparing CPU and GPU results.

### Lab 4: Image Enhancement (CLAHE) [C / CUDA]
This lab requires implementing the Contrast Limited Adaptive Histogram Equalization (CLAHE) algorithm on a GPU to improve image contrast. The implementation is divided into tiling the image, calculating histograms per tile using shared memory and atomic operations and performing bilinear interpolation to eliminate artifacts.

### Lab 5: N-Body Simulation [C / OpenMP & CUDA]
This project focuses on optimizing an N-Body simulation, which models the physical interaction of bodies (e.g., galaxies) under gravitational forces. The assignment requires an initial OpenMP parallelization followed by a highly optimized GPU implementation using techniques like tiling and streams to maximize the number of interactions calculated per second.
