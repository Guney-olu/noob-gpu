#include <cuda_runtime.h>
#include <stdio.h> 
#include <stdlib.h>

__global__ void launch_kernel(float *d_data){
    printf("Kernel sees d_data. First element is: %f\n", d_data[10]);
}

void basic_mem_ops(){
    const int N =1024*1024;
    size_t bytes = N * sizeof(float);

    //Host Memory
    float *h_data = (float*)malloc(bytes);
    
    // Device memory
    float *d_data;

    cudaMalloc(&d_data,bytes);
    
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 0.1f;
    }
    
    // Copy host to device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    launch_kernel<<<1,1>>>(d_data);
    
    // Copy device to host
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_data);
    cudaFree(d_data);

}
int main() {
    basic_mem_ops();
    printf("Program finished.\n");
    return 0;
}