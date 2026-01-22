All the cuda source files are located in the src/ folder

To compile the makefile run make

Make will create a folder named target/ that contains all the executables

Runing make clean with delete the folder target/
This project contains CUDA programs for separable 2D convolution: Convolution2D_2, Convolution2D_4, Convolution2D_6a, Convolution2D_6b, and Convolution2D_8. To build everything, run `make`. All executables are placed in the `target` directory, for example `target/Convolution2D_4`.  

Convolution2D_2 expects interactive input: (1) a filter radius r (positive integer, r ≤ 16), and (2) an image size N (power of two, with N > 2*r + 1 and N ≤ 32).  

Convolution2D_4 also expects interactive input: (1) a filter radius r (positive integer, r < 256), and (2) an initial image size N (power of two, with N > 2*r + 1 and maximum size 16,384). After receiving the input, the program automatically doubles N in a loop and prints execution timings for multiple image sizes.  

Convolution2D_6a and Convolution2D_6b do not expect any input; they run automatically using predefined parameters and print multiple results.  

To run any program, execute for example `./target/Convolution2D_4`. To clean all build artifacts, run `make clean`.
