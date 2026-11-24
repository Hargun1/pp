#include <stdio.h>
#include <omp.h>

#define N 3

int main(void) {
    int A[N][N] = {{1,2,3}, {4,5,6}, {7,8,9}};
    int B[N][N] = {{1,2,3}, {4,5,6}, {7,8,9}};
    int C[N][N] = {0};
    int thread_counts[] = {1, 2, 4};

    for (int t = 0; t < 3; t++) {
        omp_set_num_threads(thread_counts[t]);

        double start = omp_get_wtime();

        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i][j] = 0;
                for (int k = 0; k < N; k++)
                    C[i][j] += A[i][k] * B[k][j];
            }
        }

        double end = omp_get_wtime();
        printf("Threads: %d, Time: %f seconds\n",
               thread_counts[t], end - start);
    }

    printf("Result matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }

    return 0;
}
