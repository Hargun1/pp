#include <stdio.h>
#include <omp.h>

#define N 16
#define THRESHOLD 4

void merge(int a[], int temp[], int left, int mid, int right) {
    int i = left, j = mid + 1, k = left;

    while (i <= mid && j <= right)
        temp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];

    while (i <= mid) temp[k++] = a[i++];
    while (j <= right) temp[k++] = a[j++];

    for (i = left; i <= right; i++)
        a[i] = temp[i];
}

void merge_sort_seq(int a[], int temp[], int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;
    merge_sort_seq(a, temp, left, mid);
    merge_sort_seq(a, temp, mid + 1, right);
    merge(a, temp, left, mid, right);
}

void merge_sort_parallel(int a[], int temp[], int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;

    if (right - left + 1 >= THRESHOLD) {
        #pragma omp task shared(a, temp) firstprivate(left, mid)
        merge_sort_parallel(a, temp, left, mid);

        #pragma omp task shared(a, temp) firstprivate(mid, right)
        merge_sort_parallel(a, temp, mid + 1, right);

        #pragma omp taskwait
    } else {
        merge_sort_seq(a, temp, left, mid);
        merge_sort_seq(a, temp, mid + 1, right);
    }

    merge(a, temp, left, mid, right);
}

int main(void) {
    int original[N] = {8, 3, 7, 4, 2, 6, 1, 5,
                       16, 12, 14, 10, 9, 11, 15, 13};
    int seq[N], par[N], temp[N];

    for (int i = 0; i < N; i++)
        seq[i] = par[i] = original[i];

    double start = omp_get_wtime();
    merge_sort_seq(seq, temp, 0, N - 1);
    double seq_time = omp_get_wtime() - start;

    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        merge_sort_parallel(par, temp, 0, N - 1);
    }
    double par_time = omp_get_wtime() - start;

    printf("Sorted array: ");
    for (int i = 0; i < N; i++)
        printf("%d ", par[i]);

    printf("\nSequential time: %f seconds\n", seq_time);
    printf("Parallel time:   %f seconds\n", par_time);
    if (par_time > 0.0)
        printf("Speedup:         %f\n", seq_time / par_time);

    return 0;
}
