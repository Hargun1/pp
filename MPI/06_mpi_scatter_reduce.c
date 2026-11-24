#include <stdio.h>
#include <mpi.h>

#define N 8

int main(int argc, char **argv) {
    int rank, size;
    int data[N];
    int local[N];
    int local_sum = 0, total_sum = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            printf("Please run with a process count that divides %d.\n", N);
        MPI_Finalize();
        return 0;
    }

    int count = N / size;

    if (rank == 0) {
        for (int i = 0; i < N; i++)
            data[i] = i + 1;
    }

    MPI_Scatter(data, count, MPI_INT,
                local, count, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < count; i++)
        local_sum += local[i];

    printf("Process %d local sum = %d\n", rank, local_sum);

    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Total sum = %d\n", total_sum);

    MPI_Finalize();
    return 0;
}
