#include "core.h"

extern void cudaCallbackCPU(int k, int m, int n, float* searchPoints,
                            float* referencePoints, int** results)
{

    int* tmp = (int*)malloc(sizeof(int) * m);
    int minIndex;
    float minSquareSum, diff, squareSum;

    // Iterate over all search points
    for (int mInd = 0; mInd < m; mInd++) {
        minSquareSum = -1;
        // Iterate over all reference points
        for (int nInd = 0; nInd < n; nInd++) {
            squareSum = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                diff = searchPoints[k * mInd + kInd] - referencePoints[k * nInd + kInd];
                squareSum += (diff * diff);
            }
            if (minSquareSum < 0 || squareSum < minSquareSum) {
                minSquareSum = squareSum;
                minIndex = nInd;
            }
        }
        tmp[mInd] = minIndex;
    }

    *results = tmp;
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}



// searchParallel
__global__ void kernel1(int k, int m, int n, float* searchPoints,
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

extern void cudaCallback1(int k, int m, int n, float* searchPoints,
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


    kernel1 <<< block_size, 1024>>> (k, m, n, searchPoints_d, referencePoints_d, indices_d);

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


// referenceParallel
__global__ void kernel2(int k, int n, float* sub_searchPoints,
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

extern void cudaCallback2(int k, int m, int n, float* searchPoints,
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

        kernel2 <<< block_size, 1024>>> (k, n, sub_searchPoints_d, referencePoints_d, dist_d);

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


// combinedParallel
__global__ void RPkernel(int k, int n, float* sub_searchPoints,
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

__global__ void SPkernel(int k, int m, int n, float* searchPoints,
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

extern void cudaCallback3(int k, int m, int n, float* searchPoints,
                          float* referencePoints, int** results)
{
    if (m >= 1024) {
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


        SPkernel <<< block_size, 1024>>> (k, m, n, searchPoints_d, referencePoints_d, indices_d);

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
    }
    else {
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

            RPkernel <<< block_size, 1024>>> (k, n, sub_searchPoints_d, referencePoints_d, dist_d);

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
    }
    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}


// dynamicParallel
__global__ void kernel4(int k, int n, int bias, float* searchPoints,
                        float* referencePoints, float* dist)
{
    float diff, squareSum;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        squareSum = 0;
        for (int i = 0; i < k; i++) {
            diff = searchPoints[k * bias + i] - referencePoints[k * tid + i];
            squareSum += (diff * diff);
        }
        dist[bias * n + tid] = squareSum;
    }
}


__global__ void over_kernel4(int k, int m, int n, int block_size,
                             float* searchPoints, float* referencePoints,
                             float* dist_d, int* res)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        int minIndex;
        float minSquareSum;

        kernel4 <<< block_size, 1024>>> (k, n, tid, searchPoints, referencePoints, dist_d);
        
        cudaDeviceSynchronize();
        
        minSquareSum = -1;
        for (int j = 0; j < n; j++) {
            if (minSquareSum < 0 || minSquareSum > dist_d[tid * n + j]) {
                minSquareSum = dist_d[tid * n + j];
                minIndex = j;
            }
        }
        res[tid] = minIndex;
    }
}


extern void cudaCallback4(int k, int m, int n, float* searchPoints,
                          float* referencePoints, int** results)
{
    int block_size1 = divup(m, 1024);
    int block_size2 = divup(n, 1024);
    
    int* tmp_h = (int*)malloc(sizeof(int) * m);

    int* tmp_d;
    float* searchPoints_d;
    float* referencePoints_d;
    float* dist_d;

    // Memory allocation
    CHECK(cudaMalloc((void**)(&tmp_d), sizeof(int) * m));
    CHECK(cudaMalloc((void**)(&dist_d), sizeof(float) * m * n));
    CHECK(cudaMalloc((void**)(&searchPoints_d), sizeof(float) * k * m));
    CHECK(cudaMalloc((void**)(&referencePoints_d), sizeof(float) * k * n));


    // Memory Copy : Host to Device
    CHECK(cudaMemcpy((void*)(tmp_d), (void*)(tmp_h), sizeof(int) * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)(searchPoints_d), (void*)(searchPoints), sizeof(float) * k * m, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy((void*)(referencePoints_d), (void*)(referencePoints), sizeof(float) * k * n, cudaMemcpyHostToDevice));

    over_kernel4 <<< block_size1, 1024>>>(k, m, n, block_size2, searchPoints_d, referencePoints_d, dist_d, tmp_d);

    
    CHECK(cudaDeviceSynchronize());

    // Memory Copy : Device to Host
    CHECK(cudaMemcpy((void*)(tmp_h), (void*)(tmp_d), sizeof(int) * m, cudaMemcpyDeviceToHost));
    
    *results = tmp_h;
    
    cudaFree((void*)tmp_d);
    cudaFree((void*)dist_d);
    cudaFree((void*)searchPoints_d);
    cudaFree((void*)referencePoints_d);

    tmp_d = NULL;
    dist_d = NULL;
    referencePoints_d = NULL;
    searchPoints_d = NULL;

    // Note that you don't have to free searchPoints, referencePoints, and
    // *results by yourself
}