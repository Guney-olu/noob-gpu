#include <stdio.h>

#define N 10000000

__global__ void vector_add(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        c[i] = a[i] + b[i];
    }
}
int main(){
    float *h_a, *h_b, *h_out; 
    float *d_a, *d_b, *d_out; 
    int vector_size_bytes = sizeof(float) * N;

    h_a = (float*)malloc(vector_size_bytes);
    h_b = (float*)malloc(vector_size_bytes);
    h_out = (float*)malloc(vector_size_bytes);

    for(int i = 0; i < N; i++){
        h_a[i] = 1.0f; 
        h_b[i] = 2.0f;
    }

    // This Allocate device memory for a, b, and the output vector
    cudaMalloc((void**)&d_a, vector_size_bytes);
    cudaMalloc((void**)&d_b, vector_size_bytes);
    cudaMalloc((void**)&d_out, vector_size_bytes);

    // This Transfer data for a and b from host (h_) to device (d_)
    cudaMemcpy(d_a, h_a, vector_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vector_size_bytes, cudaMemcpyHostToDevice);

    // We'll use 256 threads per block.
    int threadsPerBlock = 1;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel with %d blocks of %d threads each.\n", blocksPerGrid, threadsPerBlock);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_out, N);

    cudaMemcpy(h_out, d_out, vector_size_bytes, cudaMemcpyDeviceToHost);

    printf("Verifying result...\n");
    if (h_out[123] == 3.0f) {
        printf("Success! h_out[123] is %f\n", h_out[123]);
    } else {
        printf("Failure! h_out[123] is %f, expected 3.0\n", h_out[123]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(h_a);
    free(h_b);
    free(h_out);
    return 0;
}