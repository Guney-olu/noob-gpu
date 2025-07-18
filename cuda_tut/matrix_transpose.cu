#include <cuda_runtime.h>
#include <stdio.h> 
#include <stdlib.h>
#include <iomanip>
#include <iostream>

void printMatrix(const int* matrix, int N) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            std::cout << std::setw(4) << matrix[j * N + i] << " ";
        }
        std::cout << std::endl;
    }
}

//Easy Method-1
__global__ void naive_transpose(int *input, int *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        // Non-coalesced write
        output[j * N + i] = input[i * N + j];
    }
}

//Method-2
__global__ void optimized_transpose(int *input, int *output, int N) {
    __shared__ int tile[32][32];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Coalesced read
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }
    
    __syncthreads();
    
    // Transposed indices for output
    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    
    // Coalesced write
    if (x < N && y < N) {
        output[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}


int main() {

    const int N = 2;
    
    size_t bytes = N * N * sizeof(int);

    int* h_input = new int[N * N];
    int* h_output = new int[N * N];
    
    h_input[0] = 1; h_input[1] = 2;
    h_input[2] = 4; h_input[3] = 5;

    int* d_input;
    int* d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);

    //naive_transpose<<<numBlocks, threadsPerBlock>>>(d_input,d_output,N);  
    
    optimized_transpose<<<numBlocks, threadsPerBlock>>>(d_input,d_output,N);  

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);
    
    printMatrix(h_output,N);

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Program finished.\n");
    return 0;
}