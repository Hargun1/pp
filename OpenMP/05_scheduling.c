#include <stdio.h>
#include <omp.h>

#define N 2000
#define THREADS 4

int main(void) {
    double sums[THREADS];
    int counts[THREADS];
    double start, end;

    // Static scheduling
    for (int t = 0; t < THREADS; t++) {
        sums[t] = 0.0;
        counts[t] = 0;
    }

    start = omp_get_wtime();
    #pragma omp parallel for num_threads(THREADS) schedule(static)
    for (int i = 0; i < N; i++) {
        int id = omp_get_thread_num();
        double work = 0.0;
        for (int j = 0; j < (i % 100 + 1) * 100; j++)
            work += j * 0.000001;
        sums[id] += work;
        counts[id]++;
    }
    end = omp_get_wtime();
    printf("Static time: %f seconds\n", end - start);
    for (int t = 0; t < THREADS; t++)
        printf("  Thread %d handled %d iterations\n", t, counts[t]);

    // Dynamic scheduling
    for (int t = 0; t < THREADS; t++) {
        sums[t] = 0.0;
        counts[t] = 0;
    }

    start = omp_get_wtime();
    #pragma omp parallel for num_threads(THREADS) schedule(dynamic, 10)
    for (int i = 0; i < N; i++) {
        int id = omp_get_thread_num();
        double work = 0.0;
        for (int j = 0; j < (i % 100 + 1) * 100; j++)
            work += j * 0.000001;
        sums[id] += work;
        counts[id]++;
    }
    end = omp_get_wtime();
    printf("Dynamic time: %f seconds\n", end - start);
    for (int t = 0; t < THREADS; t++)
        printf("  Thread %d handled %d iterations\n", t, counts[t]);

    // Guided scheduling
    for (int t = 0; t < THREADS; t++) {
        sums[t] = 0.0;
        counts[t] = 0;
    }

    start = omp_get_wtime();
    #pragma omp parallel for num_threads(THREADS) schedule(guided)
    for (int i = 0; i < N; i++) {
        int id = omp_get_thread_num();
        double work = 0.0;
        for (int j = 0; j < (i % 100 + 1) * 100; j++)
            work += j * 0.000001;
        sums[id] += work;
        counts[id]++;
    }
    end = omp_get_wtime();
    printf("Guided time: %f seconds\n", end - start);
    for (int t = 0; t < THREADS; t++)
        printf("  Thread %d handled %d iterations\n", t, counts[t]);

    return 0;
}
