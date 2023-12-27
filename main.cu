#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>

#define MASK_SIZE 3
#define TILE_SIZE 32

// Define size, width, and height
const int width = 1000;
const int height = 1000;
const int size = width * height;

// Function for GPU implementation
__global__ void convolutionKernel(float *inputImage, float *outputImage, int width, int height) {
    // (Same as before...)
}

// Function for CPU implementation
void convolutionCPU(float *inputImage, float *outputImage, int width, int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int startCol = col - MASK_SIZE / 2;
            int startRow = row - MASK_SIZE / 2;

            float sum = 0.0f;

            for (int i = 0; i < MASK_SIZE; ++i) {
                for (int j = 0; j < MASK_SIZE; ++j) {
                    int currentCol = startCol + j;
                    int currentRow = startRow + i;

                    if (currentCol >= 0 && currentCol < width && currentRow >= 0 && currentRow < height) {
                        sum += inputImage[currentRow * width + currentCol];
                    }
                }
            }

            outputImage[row * width + col] = sum / (MASK_SIZE * MASK_SIZE);
        }
    }
}

int main(int argc, char **argv) {
    // Check if a command-line argument is provided to choose implementation
    bool useGPU = true;
    if (argc > 1 && std::string(argv[1]) == "cpu") {
        useGPU = false;
    }

    // Allocate and initialize input image on CPU
    float inputImage[size];
    for (int i = 0; i < size; ++i) {
        inputImage[i] = static_cast<float>(rand() % 256);  // Random values between 0 and 255
    }

    // Allocate GPU memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size * sizeof(float));
    cudaMalloc((void **)&d_output, size * sizeof(float));

    // Copy input image to GPU
    cudaMemcpy(d_input, inputImage, size * sizeof(float), cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    if (useGPU) {
        // Measure time using cudaEvent for GPU implementation
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start event
        cudaEventRecord(start);

        // Launch the GPU kernel
        convolutionKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);

        // Record stop event
        cudaEventRecord(stop);

        // Synchronize and check for errors
        cudaDeviceSynchronize();
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
            return 1;
        }

        // Print GPU time
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << "Time taken by convolutionKernel (GPU): " << elapsedTime << " milliseconds" << std::endl;

    } else {
        // Measure time for CPU implementation
        clock_t startCPU = clock();

        // Run the CPU implementation
        convolutionCPU(inputImage, inputImage, width, height);

        // Measure elapsed time
        clock_t stopCPU = clock();
        double elapsedTime = static_cast<double>(stopCPU - startCPU) / CLOCKS_PER_SEC * 1000.0;
        std::cout << "Time taken by convolutionCPU (CPU): " << elapsedTime << " milliseconds" << std::endl;
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
