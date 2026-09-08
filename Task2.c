/*Matrix Multiplication with MPI (Distributed memory) */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1024

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;

    int (*A)[N] = malloc(sizeof(int[N][N]));
    int (*B)[N] = malloc(sizeof(int[N][N]));
    int (*C)[N] = malloc(sizeof(int[N][N]));
    int (*local_A)[N] = malloc(sizeof(int[rows_per_proc][N]));
    int (*local_C)[N] = calloc(rows_per_proc * N, sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A, rows_per_proc * N, MPI_INT, local_A, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                local_C[i][j] += local_A[i][k] * B[k][j];

    MPI_Gather(local_C, rows_per_proc * N, MPI_INT, C, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); 
    double end = MPI_Wtime();

    double time = end - start;
    double maxtime = 0.0;
    MPI_Reduce(&time, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
    
    if (rank == 0) {
        printf("Total time: %f s\n", maxtime);
        long long checksum = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                checksum += C[i][j];
        printf("Checksum: %lld\n", checksum);
    }

    free(A); free(B); free(C); free(local_A); free(local_C);
    MPI_Finalize();
    return 0;
}




