/* Matrix multiplication with CUDA*/
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define THREADS_PER_BLOCK 128
#define NUM_BLOCKS 1

__global__ void kernel(int *A, int *B, int *C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int totalElements = n * n;

    for (int index = tid; index < totalElements; index += stride) {
        int i = index / n;
        int j = index % n;

        int sum = 0;

        for (int k = 0; k < n; k++) {
            sum += A[i * n + k] * B[k * n + j];
        }

        C[i * n + j] = sum;
    }
}

int main() {
    int *A = (int*)malloc(N*N*sizeof(int));
    int *B = (int*)malloc(N*N*sizeof(int));
    int *C = (int*)malloc(N*N*sizeof(int));
    
    for (int i = 0; i < N*N; i++){
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(int));
    cudaMalloc(&d_B, N*N*sizeof(int));
    cudaMalloc(&d_C, N*N*sizeof(int));

    cudaMemcpy(d_A, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = NUM_BLOCKS;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start); 
    kernel<<<numBlocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop); 

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("N: %d\n", N);
    printf("Blocks: %d\n", numBlocks);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Time: %f ms\n", ms);

    cudaMemcpy(C, d_C, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    long long checksum = 0;
    for (int i = 0; i < N*N; i++) {
        checksum += C[i];
    }
    printf("Checksum: %lld\n", checksum);


    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    free(A); 
    free(B); 
    free(C);
    return 0;
}

