/*Matrix Multiplication */ 
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define N 1024

void IJKLoop(int A[N][N], int B[N][N], int C[N][N]){
    for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        C[i][j] = 0;
        for (int k = 0; k < N; k++) {
            C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
}

int main(){ 
    int (*A)[N] = malloc(sizeof(int[N][N]));
    int (*B)[N] = malloc(sizeof(int[N][N]));
    int (*C)[N] = malloc(sizeof(int[N][N]));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10; 
            B[i][j] = rand() % 10;
            C[i][j] = 0;
        }
    
        double start = omp_get_wtime();
        IJKLoop(A,B,C);
        double end = omp_get_wtime();
        printf("Time: %f\n", end - start);

        long long checksum = 0;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                checksum += C[i][j];
        printf("Checksum: %lld\n", checksum);

    free(A); free(B); free(C);
    return 0;
}