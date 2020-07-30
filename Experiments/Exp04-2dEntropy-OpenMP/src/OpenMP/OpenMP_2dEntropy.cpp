/*#########################################
##  ID: 17341137 宋震鹏
##  Introduction:
##     Calculate Entropy - OpenMP core
#########################################*/
#include "OpenMP_2dEntropy.h"

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
##  - idx: index in 1d format.
##  - rows: number of rows in input.
##  - cols: number of columns in input.
##  - CACHED_LOG: Flag to set whether to
##                use cached log result.
#########################################*/
void calculateSubEntropy(float* input, float* output, float* logres, int idx, int rows, int cols, bool CACHED_LOG)
{
    // transfer index in 1d format to 2d format.
    int row_id = idx / cols;
    int col_id = idx % cols;
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
                subarr[int(input[idx])]++;
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
##  - PARALLEL: Flag to set whether to
##                use OpenMP.
##  - CACHED_LOG: Flag to set whether to
##                use cached log result.
#########################################*/
void calculateEntropy(float* input, float* output, float* logres, int rows, int cols, bool PARALLEL, bool CACHED_LOG)
{
    if (!PARALLEL) {
        printf("OpenMP is not active.\nUse `./main 1 [cached log(0/1)] [verbose(0/1)]` to turn on OpenMP.\n");
        for (int i = 0; i < rows * cols; i++) {
            calculateSubEntropy(input, output, logres, i, rows, cols, CACHED_LOG);
        }
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < rows * cols; i++) {
            calculateSubEntropy(input, output, logres, i, rows, cols, CACHED_LOG);
        }
    }
}
