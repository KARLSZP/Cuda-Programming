/*#########################################
##  ID: 17341137 宋震鹏
##  Introduction: Calculate Entropy - main
#########################################*/
#include "cuda/cuda_2dEntropy.cuh"
#include "utils/samples.h"
#include "utils/utils.h"

std::string dataPath = "../test/data.bin";
std::string outputPath = "../out/data.out";


/*#########################################
##  @Function: main
##  @Description: main function
##=========================================
##  @Parameters:
##  - argc: number of arguments passed.
##  - argv: list of arguments passed.
#########################################*/
int main(int argc, char const* argv[])
{
    /* 
     * Check if cmd arguments are valid.
     * Note:
     *   1. Usage: ./main [shared memory(0/1)] [cached log(0/1)] [verbose(0/1)]
     *   2. 0 is always for false/off, 1 is always for true/on.
     *   3. arguments are read in one-by-one, default value only work for 
     *      arguments at the back.
     */
    if (argc > 4) {
        printf("Invalid Usages.\n");
        printf("Usage: ./main [shared memory(0/1)] [cached log(0/1)] [verbose(0/1)]\n");
        exit(-1);
    }
    const int printSample = argc == 4 ? atoi(argv[3]) : 1;
    const int printResult = argc == 4 ? atoi(argv[3]) : 1;
    const int SHARED_MEM = argc > 1 ? atoi(argv[1]) : 0;
    const int CACHED_LOG = argc > 2 ? atoi(argv[2]) : 0;

    // Open the data file
    FILE* instream = fopen(dataPath.c_str(), "rb");
    if (instream == NULL) {
        printf("failed to open the data file\n");
        return -1;
    }

    // Open a stream to write out results in text
    FILE* outStream = fopen(outputPath.c_str(), "w");
    if (outStream == NULL) {
        printf("failed to open the output file\n");
        return -1;
    }

    // Initialize variables
    int ROWS, COLS;
    long t1, t2;
    double time_in_total = 0.0;

    /*
     * VAR_h : host variable.
     * VAR_d : device variable.
     */
    float* input_h;
    float* input_d;
    float* output_h;
    float* output_d;
    float* logres_h;
    float* logres_d;

    // Cached log result
    logres_h = (float*)malloc(26 * sizeof(float));
    for (int i = 1; i <= 25; i++) {
        logres_h[i] = log(i);
    }

    // Memory allocation
    CHECK(cudaMalloc((void**)(&logres_d), 26 * sizeof(float)));
    // Memory Copy : Host to Device
    CHECK(cudaMemcpy((void*)(logres_d), (void*)(logres_h), 26 * sizeof(float), cudaMemcpyHostToDevice));

    // Read in and process the samples one-by-one
    while (getNextSample(instream, &COLS, &ROWS, &input_h) != 0) {

        output_h = (float*)malloc(ROWS * COLS * sizeof(float));

        // Verbose
        if (printSample) {
            // Print out a small portion of the sample
            printf("\nsample: [%d x %d]\n", ROWS, COLS);
            for (int j = ROWS - 5; j < ROWS; j++) {
                for (int i = COLS - 5; i < COLS; i++) {
                    printf("%8.5f ", input_h[j * COLS + i]);
                }
                printf("\n");
            }
        }


        // Memory allocation
        CHECK(cudaMalloc((void**)(&input_d), ROWS * COLS * sizeof(int)));
        CHECK(cudaMalloc((void**)(&output_d), ROWS * COLS * sizeof(float)));

        // Memory Copy : Host to Device
        CHECK(cudaMemcpy((void*)(input_d), (void*)(input_h), ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy((void*)(output_d), (void*)(output_h), ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice));

        // Time record
        t1 = getTime();
        
        // Call kernel
        calculateEntropy(input_d, output_d, logres_d, ROWS, COLS, SHARED_MEM, CACHED_LOG);

        // Time record
        t2 = getTime();

        // Memory Copy : Device to Host
        CHECK(cudaMemcpy((void*)(input_h), (void*)(input_d), ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy((void*)(output_h), (void*)(output_d), ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost));

        // Verbose
        if (printResult) {
            // Print out a small portion of the result
            printf("result:\n");
            for (int j = ROWS - 5; j < ROWS; j++) {
                for (int i = COLS - 5; i < COLS; i++) {
                    printf("%8.5f ", output_h[j * COLS + i]);
                }
                printf("\n");
            }
        }

        // Report - sample
        time_in_total += double(t2 - t1) / 1000000000L;
        printf("Calculation is done in %f second(s).\n", double(t2 - t1) / 1000000000L);


        // Write the result to the output stream
        char buffer[128];
        sprintf(buffer, "%d,", COLS);
        W_CHK(fputs(buffer, outStream));
        sprintf(buffer, "%d,", ROWS);
        W_CHK(fputs(buffer, outStream));
        for (int i = 0; i < ROWS * COLS; i++) {
            sprintf(buffer, "%.5f,", output_h[i]);
            W_CHK(fputs(buffer, outStream));
        }
        W_CHK(fputs("\n", outStream));


        // De-allocate the sample and the result
        cudaFree((void*)input_d);
        cudaFree((void*)output_d);

        free(input_h);
        free(output_h);
        input_h = NULL;
        output_h = NULL;
    }

    // Report - ALL
    printf("\n>>> Samples were all done in %f second(s).\n", time_in_total);
    printf(">>> Shared Memory: %s\n", SHARED_MEM ? "ON" : "OFF");
    printf(">>> Cached Log(0~25): %s\n", CACHED_LOG ? "ON" : "OFF");

    cudaFree((void*)logres_d);
    free(logres_h);
    logres_d = NULL;
    logres_h = NULL;


    // Close the output stream
    fclose(instream);
    fclose(outStream);
    return 0;
}