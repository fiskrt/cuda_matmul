
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <iostream>

//#include "matmul.cu"

__global__ void matmul_kernel(
    float *C, const float *A, const float *B,
    int M, int N, int K
);

void create_tensors(float **A, float **B, float **C, float **C_ref, int M, int N, int K) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    *A = (float*)malloc(size_A);
    *B = (float*)malloc(size_B);
    *C = (float*)malloc(size_C);
    *C_ref = (float*)malloc(size_C);
    
    srand(time(NULL));
    
    for (int i = 0; i < M * K; i++) {
        (*A)[i] = (float)rand() / RAND_MAX;
    }
    
    for (int i = 0; i < K * N; i++) {
        (*B)[i] = (float)rand() / RAND_MAX;
    }
    
    for (int i = 0; i < M * N; i++) {
        (*C)[i] = 0.0f;
        (*C_ref)[i] = 0.0f;
    }
}

void cpu_matmul(float *C, const float *A, const float *B, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool verify_result(const float *C_gpu, const float *C_cpu, int M, int N) {
    const float tolerance = 1e-3f;
    
    for (int i = 0; i < M * N; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
}


void get_sm_version() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        int major = prop.major;
        int minor = prop.minor;

        std::cout << "GPU " << i << ": " << prop.name
                  << ", Compute Capability: " << major << "." << minor << "\n";
    }
}


void run(int block_size) {
    const int M = 4096;
    const int K = 10240;
    const int N = 4096; 
    
    printf("Multiplication: C(%dx%d) = A(%dx%d) * B(%dx%d)\n", M, N, M, K, K, N);
    get_sm_version();
    
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;
    
    create_tensors(&h_A, &h_B, &h_C, &h_C_ref, M, N, K);
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 blockSize(block_size, block_size);
    dim3 gridSize((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_kernel<<<gridSize, blockSize>>>(d_C, d_A, d_B, M, N, K);
    cudaEventRecord(stop);
    // Get any errors from kernel, e.g. too large block size...
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR!: CUDA kernel error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    printf("GPU Time: %.2f ms\n", gpu_time);
    if (M <= 1024 && N <= 1024 && K <= 1024) {
        clock_t cpu_start = clock();
        cpu_matmul(h_C_ref, h_A, h_B, M, N, K);
        clock_t cpu_end = clock();
        float cpu_time = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;
        
        printf("CPU Time: %.2f ms\n", cpu_time);
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
        
        if (verify_result(h_C, h_C_ref, M, N)) {
            printf("Results match!\n");
        } else {
            printf("Results do not match!\n");
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {
    run(32);
    //for (int block_size : {4, 8, 16, 24, 25, 32, 40, 64}) {
    //    printf("\nRunning with block size %d...\n", block_size);
    //    run(block_size);
    //}
    return 0;
}