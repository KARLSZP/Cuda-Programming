#include "core.h"


__global__ void kernel(int k, int m, int n, float* searchPoints,
                       float* referencePoints, int* indices)
{
    int minIndex;
    float minSquareSum, diff, squareSum;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        minSquareSum = -1;
        // Iterate over all reference points
        for (int nInd = 0; nInd < n; nInd++) {
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                diff = searchPoints[k * tid + kInd] - referencePoints[k * nInd + kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        indices[tid] = minIndex;
    }
}

extern void cudaCallback(int k, int m, int n, float* searchPoints,
                         float* referencePoints, int** results)
{

    int block_size = divup(m, 1024);

    int* indices_h = (int*)malloc(sizeof(int) * m);

    int* indices_d;
    float* searchPoints_d;
    float* referencePoints_d;

    // Memory allocation
    CHECK(cudaMalloc((void**)(&indices_d), sizeof(int) * m));
    CHECK(cudaMalloc((void**)(&searchPoints_d), sizeof(float) * k * m));
    CHECK(cudaMalloc((void**)(&referencePoints_d), sizeof(float) * k * n));

    // Memory Copy : Host to Device
    CHECK(cudaMemcpy((void*)(indices_d), (void*)(indices_h), sizeof(int) * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)(searchPoints_d), (void*)(searchPoints), sizeof(float) * k * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)(referencePoints_d), (void*)(referencePoints), sizeof(float) * k * n, cudaMemcpyHostToDevice));


    kernel <<< block_size, 1024>>> (k, m, n, searchPoints_d, referencePoints_d, indices_d);

    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy((void*)(indices_h), (void*)(indices_d), sizeof(int) * m, cudaMemcpyDeviceToHost));


    *results = indices_h;

    cudaFree((void*)indices_d);
    cudaFree((void*)searchPoints_d);
    cudaFree((void*)referencePoints_d);

    indices_h = NULL;
    indices_d = NULL;
    searchPoints_d = NULL;
    referencePoints_d = NULL;

    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}
