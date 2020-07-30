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


__global__ void over_kernel(int k, int m, int n, int block_size,
                            float* searchPoints, float* referencePoints,
                            float* dist_d, float* sub_searchPoints_d, int* res)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {

        int minIndex;
        float minSquareSum;

        for (int j = 0; j < k; j++) {
            sub_searchPoints_d[j] = searchPoints[tid * k + j];
        }

        kernel <<< block_size, 1024>>> (k, n, sub_searchPoints_d, referencePoints, dist_d);

        minSquareSum = -1;
        for (int j = 0; j < n; j++) {
            if (minSquareSum < 0 || minSquareSum > dist_d[j]) {
                minSquareSum = dist_d[j];
                minIndex = j;
            }
        }

        res[tid] = minIndex;


    }

}


extern void cudaCallback(int k, int m, int n, float* searchPoints,
                         float* referencePoints, int** results)
{
    int block_size1 = divup(m, 1024);
    int block_size2 = divup(n, 1024);

    int* tmp_h = (int*)malloc(sizeof(int) * m);

    int* tmp_d;
    float* dist_d;
    float* searchPoints_d;
    float* referencePoints_d;
    float* sub_searchPoints_d;

    // Memory allocation
    CHECK(cudaMalloc((void**)(&tmp_d), sizeof(int) * m));
    CHECK(cudaMalloc((void**)(&searchPoints_d), sizeof(float) * k * m));
    CHECK(cudaMalloc((void**)(&referencePoints_d), sizeof(float) * k * n));

    CHECK(cudaMalloc((void**)(&dist_d), sizeof(float) * n));
    CHECK(cudaMalloc((void**)(&sub_searchPoints_d), sizeof(float) * k));

    // Memory Copy : Host to Device
    CHECK(cudaMemcpy((void*)(tmp_d), (void*)(tmp_h), sizeof(int) * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)(searchPoints_d), (void*)(searchPoints), sizeof(float) * k * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)(referencePoints_d), (void*)(referencePoints), sizeof(float) * k * n, cudaMemcpyHostToDevice));

    over_kernel <<< block_size1, 1024>>>(k, m, n, block_size2, searchPoints_d, referencePoints_d, dist_d, sub_searchPoints_d, tmp_d);

    CHECK(cudaDeviceSynchronize());

    // Memory Copy : Device to Host
    CHECK(cudaMemcpy((void*)(tmp_h), (void*)(tmp_d), sizeof(int) * m, cudaMemcpyDeviceToHost));


    *results = tmp_h;

    cudaFree((void*)tmp_d);
    cudaFree((void*)dist_d);
    cudaFree((void*)searchPoints_d);
    cudaFree((void*)referencePoints_d);
    cudaFree((void*)sub_searchPoints_d);


    tmp_d = NULL;
    dist_d = NULL;
    referencePoints_d = NULL;
    searchPoints_d = NULL;
    sub_searchPoints_d = NULL;

    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}