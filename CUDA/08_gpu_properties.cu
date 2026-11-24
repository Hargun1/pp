#include <stdio.h>
#include <cuda_runtime.h>

__global__ void show_threads(void) {
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    if (err != cudaSuccess || count == 0) {
        printf("No CUDA-capable GPU found.\n");
        return 1;
    }

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("GPU name: %s\n", p.name);
    printf("Global memory: %zu MB\n",
           (size_t)(p.totalGlobalMem / (1024 * 1024)));
    printf("Multiprocessors: %d\n", p.multiProcessorCount);
    printf("Maximum threads per block: %d\n", p.maxThreadsPerBlock);

    printf("\nLaunch 1: 1 block, 4 threads\n");
    show_threads<<<1, 4>>>();
    cudaDeviceSynchronize();

    printf("\nLaunch 2: 2 blocks, 4 threads each\n");
    show_threads<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("\nLaunch 3: 2 blocks, 8 threads each\n");
    show_threads<<<2, 8>>>();
    cudaDeviceSynchronize();

    return 0;
}
