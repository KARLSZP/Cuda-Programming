#include "core.h"


__global__ void kernel(int k, int n, float* sub_searchPoints,
                       float* referencePoints, float* dist)
{
    float diff, squareSum;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        squareSum = 0;
        for (int i = 0; i < k; i++) {
            diff = sub_searchPoints[i] - referencePoints[k * tid + i];
            squareSum += (diff * diff);
        }
        dist[tid] = squareSum;
    }
}

extern void cudaCallback(int k, int m, int n, float* searchPoints,
                         float* referencePoints, int** results)
{
    int minIndex;
    float minSquareSum;

    int block_size = divup(n, 1024);

    int* tmp = (int*)malloc(sizeof(int) * m);
    float* dist_h = (float*)malloc(sizeof(float) * n);
    float* sub_searchPoints_h = (float*)malloc(sizeof(float) * k);

    float* referencePoints_d;
    float* dist_d;
    float* sub_searchPoints_d;

    // Memory allocation
    CHECK(cudaMalloc((void**)(&referencePoints_d), sizeof(float) * k * n));
    CHECK(cudaMalloc((void**)(&dist_d), sizeof(float) * n));
    CHECK(cudaMalloc((void**)(&sub_searchPoints_d), sizeof(float) * k));

    // Memory Copy : Host to Device
    CHECK(cudaMemcpy((void*)(referencePoints_d), (void*)(referencePoints), sizeof(float) * k * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)(dist_d), (void*)(dist_h), sizeof(float) * n, cudaMemcpyHostToDevice));


    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            sub_searchPoints_h[j] = searchPoints[i * k + j];
            // printf("%f ", sub_searchPoints_h[j]);
        }
        // getchar();

        // Memory Copy : Host to Device
        CHECK(cudaMemcpy((void*)(sub_searchPoints_d), (void*)(sub_searchPoints_h), sizeof(float) * k, cudaMemcpyHostToDevice));

        kernel <<< block_size, 1024>>> (k, n, sub_searchPoints_d, referencePoints_d, dist_d);

        CHECK(cudaDeviceSynchronize());

        // Memory Copy : Device to Host
        CHECK(cudaMemcpy((void*)(dist_h), (void*)(dist_d), sizeof(float) * n, cudaMemcpyDeviceToHost));

        minSquareSum = -1;
        for (int j = 0; j < n; j++) {
            if (minSquareSum < 0 || minSquareSum > dist_h[j]) {
                minSquareSum = dist_h[j];
                minIndex = j;
            }
        }

        tmp[i] = minIndex;
    }

    *results = tmp;

    free(dist_h);
    free(sub_searchPoints_h);

    cudaFree((void*)referencePoints_d);
    cudaFree((void*)dist_d);
    cudaFree((void*)sub_searchPoints_d);

    dist_h = NULL;
    sub_searchPoints_h = NULL;
    referencePoints_d = NULL;
    dist_d = NULL;
    sub_searchPoints_d = NULL;

    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}