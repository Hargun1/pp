#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1000000

__global__ void vector_add(const int *a, const int *b, int *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

int main(void) {
    int *a = (int *)malloc(N * sizeof(int));
    int *b = (int *)malloc(N * sizeof(int));
    int *cpu = (int *)malloc(N * sizeof(int));
    int *gpu = (int *)malloc(N * sizeof(int));
    int *da, *db, *dc;

    if (!a || !b || !cpu || !gpu) {
        printf("Host memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    clock_t start = clock();
    for (int i = 0; i < N; i++)
        cpu[i] = a[i] + b[i];
    double cpu_ms = 1000.0 * (clock() - start) / CLOCKS_PER_SEC;

    cudaMalloc((void **)&da, N * sizeof(int));
    cudaMalloc((void **)&db, N * sizeof(int));
    cudaMalloc((void **)&dc, N * sizeof(int));

    cudaMemcpy(da, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t begin, finish;
    cudaEventCreate(&begin);
    cudaEventCreate(&finish);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(begin);
    vector_add<<<blocks, threads>>>(da, db, dc);
    cudaEventRecord(finish);
    cudaEventSynchronize(finish);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, begin, finish);

    cudaMemcpy(gpu, dc, N * sizeof(int), cudaMemcpyDeviceToHost);

    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (cpu[i] != gpu[i]) {
            correct = 0;
            break;
        }
    }

    printf("CPU time: %f ms\n", cpu_ms);
    printf("GPU kernel time: %f ms (memory copies excluded)\n", gpu_ms);
    printf("C[10] = %d\n", gpu[10]);
    printf("Results match: %s\n", correct ? "YES" : "NO");

    cudaEventDestroy(begin);
    cudaEventDestroy(finish);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(cpu);
    free(gpu);

    return correct ? 0 : 1;
}
