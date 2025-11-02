#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Julia Set parameters
#define WIDTH 4096
#define HEIGHT 4096
#define MAX_ITER 1000

//  Complex number for julia set constant
#define C_REAL 0.0
#define C_IMAG 1.0

// CUDA kernel for Julia Set computation
__global__ void julia_set_kernel(unsigned char *output, int width, int height, float c_real, float c_imag, int max_iter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Map pixel coordinates to compex plane
    float real = (x - width / 2.0f) * 4.0f / width;
    float imag = (y - height / 2.0f) * 4.0f / height;

    // Julia set iteration
    int iter = 0;
    float z_real = real;
    float z_imag = imag;

    while (iter < max_iter && (z_real * z_real + z_imag * z_imag) < 4.0f) {
        float temp = z_real * z_real - z_imag * z_imag + c_real;
        z_imag = 2.0f * z_real * z_imag + c_imag;
        z_real = temp;
        iter++;
    } 

    // Color mapping basefd on iteration count
    int idx = y * width + x;
    if (iter == max_iter) {
        output[idx] = 0; //Inside set - black
    } else {
        // Smooth coloring
        output[idx] = (unsigned char)(255 * iter / max_iter);
    }
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do {  \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d,%s\n", __FILE__,__LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(int argc, char *argv[]) {
    printf("=== Julia Set CUDA Generator ===\n");
    printf("Resolution: %dx%d pixels\n", WIDTH, HEIGHT );
    printf("Max iterations: %d\n", MAX_ITER);
    printf("Julia constant: %.3f + %.3fi\n\n", C_REAL, C_IMAG);

    // Allocate host memory
    size_t image_size = WIDTH * HEIGHT * sizeof(unsigned char);
    unsigned char *h_output = (unsigned char*)malloc(image_size);
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    unsigned char *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, image_size));

    // Configure kernel launch parameters
    dim3 block_size(16, 16);
    dim3 grid_size((WIDTH + block_size.x -1) / block_size.x, (HEIGHT + block_size.y -1) / block_size.y);

    printf("Grid size: (%d, %d)\n", grid_size.x, grid_size.y);
    printf("Block size: (%d, %d)\n", block_size.x, block_size.y);
    printf("Total threads: %ld\n\n", (long)grid_size.x * grid_size.y * block_size.x * block_size.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup run
    printf("Performing warmup run...\n");
    julia_set_kernel<<<grid_size, block_size>>>(d_output, WIDTH, HEIGHT, C_REAL, C_IMAG, MAX_ITER);
    CUDA_CHECK(cudaDeviceSynchronize());
        
    // Benchmark runs
    const int num_runs = 10;
    float total_time = 0.0f;
    
    printf("Running %d benchmark iterations...\n", num_runs);
    
    for (int run = 0; run < num_runs; run++) {
        CUDA_CHECK(cudaEventRecord(start));
        
        julia_set_kernel<<<grid_size, block_size>>>(d_output, WIDTH, HEIGHT, C_REAL, C_IMAG, MAX_ITER);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time += milliseconds;
        
        printf("  Run %2d: %.3f ms\n", run + 1, milliseconds);
    }
    
    float avg_time = total_time / num_runs;

    // Copy resultsto host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));

    // Calculate performance metrics
    long long total_pixels = (long long)WIDTH * HEIGHT;
    long long total_iterations = total_pixels * MAX_ITER; // maximum possible
    double pixels_per_sec = (total_pixels / avg_time) * 1000.0;
    double gflops = (total_iterations * 10.0 / avg_time) / 1e9; // ~10 ops per iteration

    printf("\n=== Performance Results ===\n");
    printf("Average computation time: %.3f ms\n", avg_time);
    printf("Pixels computed: %lld\n", total_pixels);
    printf("Throughput: %.2f Mpixels/sec\n", pixels_per_sec / 1e6);
    printf("Estimated GFLOPS: %.2f\n", gflops);

    // Save results to PGM file (simple grayscale format)
    printf("\nSaving result to julia_set.pgm...\n");
    FILE *fp = fopen("julia_set.pgm", "wb");
    if (fp) {
        fprintf(fp, "P5\n%d, %d\n255\n", WIDTH, HEIGHT);
        fwrite(h_output, 1, image_size, fp);
        fclose(fp);
        printf("Image saved successfully!\n");
        printf("View with: display julia_set.pgm (ImageMagick)\n");
        printf("  or: convert julia_set.pgm julia_set.png\n");
    } else {
        fprintf(stderr, "Failed to save image file\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_output));
    free(h_output);

    printf("\nDone!\n");

    return EXIT_SUCCESS;    
}