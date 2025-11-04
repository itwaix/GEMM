// gemm_naive.cu
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 16

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    // using Shared Memory
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0;
    const int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        // A子块：行=row，列范围=[t*TILE, t*TILE+TILE)
        int aCol = t * TILE_WIDTH + tx;
        if (row < M && aCol < K)
            As[ty][tx] = A[row * K + aCol];
        else
            As[ty][tx] = 0.0f;

        // B子块：列=col，行范围=[t*TILE, t*TILE+TILE)
        int bRow = t * TILE_WIDTH + ty;
        if (bRow < K && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // 在共享内存中做一个 TILE_WIDTH 次的累加
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

    // if (row < M && col < N) {
    //     float sum = 0.0f;
    //     for (int k = 0; k < K; ++k) {
    //         sum += A[row * K + k] * B[k * N + col];
    //     }
    //     C[row * N + col] = sum;
    // }
}

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int TILE = 16;

    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    std::vector<float> hA(M * K), hB(K * N), hC(M * N), hC_ref(M * N);

    // 初始化随机数据
    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (auto &x : hA) x = dis(gen);
    for (auto &x : hB) x = dis(gen);


    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, sizeof(float) * M * K);
    cudaMalloc(&dB, sizeof(float) * K * N);
    cudaMalloc(&dC, sizeof(float) * M * N);


    cudaMemcpy(dA, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice); //使用data
    cudaMemcpy(dB, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice);
    gemm_naive_kernel<<<gridDim, blockDim>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(hC.data(), dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += hA[i * K + k] * hB[k * N + j];
            hC_ref[i * N + j] = sum;
        }
    }

    // 计算误差
    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        max_err = std::max(max_err, std::fabs(hC_ref[i] - hC[i]));
    }
    std::cout << "Max error: " << max_err << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    std::cout << "Done." << std::endl;
    return 0;
}
