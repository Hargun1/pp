#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, size;
    int data[3] = {10, 20, 30};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            printf("Run with at least 2 processes.\n");
        MPI_Finalize();
        return 0;
    }

    // Point-to-point communication: process 0 sends to process 1.
    if (rank == 0) {
        MPI_Send(data, 3, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Process 0 sent array to process 1.\n");
    } else if (rank == 1) {
        int received[3];
        MPI_Recv(received, 3, MPI_INT, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received: %d %d %d\n",
               received[0], received[1], received[2]);
    }

    // Collective communication: every process calls MPI_Bcast.
    if (rank == 0) {
        data[0] = 100;
        data[1] = 200;
        data[2] = 300;
    }

    MPI_Bcast(data, 3, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d broadcast data: %d %d %d\n",
           rank, data[0], data[1], data[2]);

    MPI_Finalize();
    return 0;
}
