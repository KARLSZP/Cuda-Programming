/*#########################################
##  ID: 17341137 宋震鹏
##  Introduction:
##     Calculate Entropy - core
#########################################*/
#include "cuda_2dEntropy.cuh"

/*#########################################
##  @Function: calculateSubEntropy
##  @Description: calculate a sub-area of 
##    input matrix, size of  sub-area  is 
##    (WINDOW_SIZE x WINDOW_SIZE) defined
##    in header  file,  not  using shared
##    memory.
##=========================================
##  @Parameters:
##  - input: input matrix, in 1d version.
##    *Note:
##      use 1d array to store 2d data, 
##      2d[i][j] = 1d[i*cols + j].
##  - output: output matrix, in 1d version,
##            each element records entropy
##            of  the  sub-area  which  is
##            centered  with  it, overflow 
##            is ignored.
##  - logres: cached result of log(1) to 
##            log(25).
##  - rows: number of rows in input.
##  - cols: number of columns in input.
##  - CACHED_LOG: Flag to set whether to
##                use cached log result.
#########################################*/
__global__ void calculateSubEntropy(float* input, float* output, float* logres, int rows, int cols, bool CACHED_LOG)
{
    /* tid : thread id, note that:
     * ====================================================
     * |  blockDim's shape:  |  normal 2d-array's shape:  |
     * |         x           |             y              |
     * |      ------>        |          ------>           |
     * |      |              |          |                 |
     * |    y |              |        x |                 |
     * |      v              |          v                 |
     * ====================================================
     * 
     * Suppose that:
     * - input : SIZE = rows * cols
     * - A block's shape is set as: (blockDim.x, 1, 1)
     * - SIZE is distributed into K blocks.
     * 
     * The Distribution could be displayed as below:
     *
     *   input              input(SIZE)
     *                   /       |      \
     *                  /        |       \
     *   blocks        1        ...       K
     *               / | \              / | \
     *   threads    1 ... blockDim.x   1 ... blockDim.x
     * 
     * So, we know that:
     * - blockDim.x : a certain value, 
     *                in this case is set to 1024.
     * - blockIdx.x : range from 1 to K 
     *                (actually is 0 to K-1).
     * - threadIdx.x : range from 1 to blockDim.x
     *                 (actually is 0 to blockDim.x-1).
     * 
     * That's why tid = = blockDim.x * blockIdx.x + threadIdx.x.
     */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Now, the element at coord.(row_id, col_id) is the centeral
    // element of the sub-area, which is going to be processed by
    // current thread.
    int row_id = tid / cols;
    int col_id = tid % cols;

    // subarr[i](i from 0 to MAX_ELEM) records 
    // how many i(s) are there in the sub-area.
    int subarr[MAX_ELEM + 1] = {0};

    double res = 0.0;

    // In case of some not divisible(by blockdim.x) input size,
    // an addtional block is dispatched for completeness, this
    // if() is used to check if coord.(row_id, col_id) is valid.
    if (row_id < rows && col_id < cols) {

        // bias determine how many elements should be taken into
        // account, around coord.(row_id, col_id) both vertically
        // and horizontally.
        int bias = WINDOW_SIZE / 2;
        /*
         * Here's two examples to show how a sub-area is selected:
         * 
         *   1. No Overflow              2. Overflowed
         *              
         *   x . x x x x x x . x         x x .  x x x x x
         *   . . .  u p . .  . .         x 1 .  2 3 4 5   
         *   . .-------------. .         . . .  . u p . . 
         *   x  | 1 2 3 4 5 |r x         . . .-----------r
         *   x l| 1 2 3 4 5 |i x         x 1 l| 2 3 4 5 |i
         *   x e| 1 2 C 4 5 |g x         x 1 e| 2 3 4 5 |g 
         *   x f| 1 2 3 4 5 |h x         x 1 f| 2 3 C 5 |h 
         *   x t| 1 2 3 4 5 |t x         x 1 t| 2 3 4 5 |t 
         *   . .-------------. .              ----------- 
         *   . . . d o w n . . .                d o w n 
         *   x . x x x x x x . x         
         * 
         *            * C is the center of the sub-area.
         *            * x is other element in input.
         */
        int up = row_id - bias > 0 ? row_id - bias : 0;
        int down = row_id + bias < rows ? row_id + bias : rows - 1;
        int left = col_id - bias > 0 ? col_id - bias : 0;
        int right = col_id + bias < cols ? col_id + bias : cols - 1;
        for (int i = up; i <= down; i++) {
            for (int j = left; j <= right; j++) {
                int idx = i * cols + j;
                subarr[int(input[idx])]++;
            }
        }
        double size = (down - up + 1) * (right - left + 1);
        for (int i = 0; i <= MAX_ELEM; i++) {
            // Calculate entropy:
            // Ent = sum{pi * log(pi)}(i from 0 to MAX_ELEM)
            if (subarr[i]) {
                double pi = subarr[i] / size;
                if (CACHED_LOG) {
                    res += -pi * (logres[subarr[i]] - logres[int(size)]);
                }
                else {
                    res += -pi * log(pi);
                }
            }
        }
        output[row_id * cols + col_id] = res;
    }
}

/*#########################################
##  @Function: 
##    calculateSubEntropy_sharedMem
##  @Description: calculate a sub-area of 
##    input matrix, size of  sub-area  is 
##    (WINDOW_SIZE x WINDOW_SIZE) defined
##    in header file, using shared memory.
##=========================================
##  @Parameters:
##  - input: input matrix, in 1d version.
##    *Note:
##      use 1d array to store 2d data, 
##      2d[i][j] = 1d[i*cols + j].
##  - output: output matrix, in 1d version,
##            each element records entropy
##            of  the  sub-area  which  is
##            centered  with  it, overflow 
##            is ignored.
##  - logres: cached result of log(1) to 
##            log(25).
##  - rows: number of rows in input.
##  - cols: number of columns in input.
##  - CACHED_LOG: Flag to set whether to
##                use cached log result.
#########################################*/
__global__ void calculateSubEntropy_sharedMem(float* input, float* output, float* logres, int rows, int cols, bool CACHED_LOG)
{
    /*
     * To accelerate, move input into shared memory.
     * For dynamic allocation, add `extern` in front of
     * __shared__.
     */
    extern __shared__ float shared_input[];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            shared_input[i*cols + j] = input[i*cols + j];
        }
    }

    // synchronization
    __syncthreads();
    

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int row_id = tid / cols;
    int col_id = tid % cols;
    int subarr[MAX_ELEM + 1] = {0};
    double res = 0.0;
    if (row_id < rows && col_id < cols) {
        int bias = WINDOW_SIZE / 2;
        int up = row_id - bias > 0 ? row_id - bias : 0;
        int down = row_id + bias < rows ? row_id + bias : rows - 1;
        int left = col_id - bias > 0 ? col_id - bias : 0;
        int right = col_id + bias < cols ? col_id + bias : cols - 1;
        for (int i = up; i <= down; i++) {
            for (int j = left; j <= right; j++) {
                int idx = i * cols + j;
                subarr[int(shared_input[idx])]++;
            }
        }
        double size = (down - up + 1) * (right - left + 1);
        for (int i = 0; i <= MAX_ELEM; i++) {
            if (subarr[i]) {
                double pi = subarr[i] / size;
                if (CACHED_LOG) {
                    res += -pi * (logres[subarr[i]] - logres[int(size)]);
                }
                else {
                    res += -pi * log(pi);
                }
            }
        }
        output[row_id * cols + col_id] = res;
    }
}

/*#########################################
##  @Function: calculateEntropy
##  @Description: Control and invoke the
##    kernel function.
##=========================================
##  @Parameters:
##  - input: input matrix, in 1d version.
##    *Note:
##      use 1d array to store 2d data, 
##      2d[i][j] = 1d[i*cols + j].
##  - output: output matrix, in 1d version,
##            each element records entropy
##            of  the  sub-area  which  is
##            centered  with  it, overflow 
##            is ignored.
##  - logres: cached result of log(1) to 
##            log(25).
##  - rows: number of rows in input.
##  - cols: number of columns in input.
##  - SHARED_MEM: Flag to set whether to
##                use shared memory.
##  - CACHED_LOG: Flag to set whether to
##                use cached log result.
#########################################*/
void calculateEntropy(float* input, float* output, float* logres, int rows, int cols, bool SHARED_MEM, bool CACHED_LOG)
{
    // In case of some not divisible(by blockdim.x) input size,
    // an addtional block is dispatched for completeness.
    int block_size = (rows * cols) % 1024 ? (rows * cols) / 1024 + 1: (rows * cols) / 1024;

    // Not using shared memory
    if (!SHARED_MEM) {
        calculateSubEntropy<<<block_size, 1024>>>(input, output, logres, rows, cols, CACHED_LOG);
    }
    // Failed in using shared memory, 
    // since shared memory is limited to a small value.
    else if (SHARED_MEM && (rows * cols) * sizeof(float) > 49152) {
        printf("\nAllocating size overflow, allocating %d bytes > shared memory size %d bytes\n", int((rows * cols) * sizeof(float)), 49152);
        printf("Use Non-shared-memory mode instead.\n\n");
        calculateSubEntropy<<<block_size, 1024>>>(input, output, logres, rows, cols, CACHED_LOG);
    }
    // Using shared memory
    else if (SHARED_MEM && (rows * cols) * sizeof(float) <= 49152) {
        const size_t shared_mem_size = (rows * cols) * sizeof(float);
        calculateSubEntropy_sharedMem<<<block_size, 1024, shared_mem_size>>>(input, output, logres, rows, cols, CACHED_LOG);
    }
    // Synchronization
    CHECK(cudaDeviceSynchronize());
}
