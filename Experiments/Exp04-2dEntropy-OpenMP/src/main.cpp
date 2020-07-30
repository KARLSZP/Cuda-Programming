#include <iostream>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <sys/time.h>
#include <unistd.h>
#include <cmath>
#include <omp.h>
#include "OpenMP/OpenMP_2dEntropy.h"
#include "utils/samples.h"
#include "utils/utils.h"

using namespace std;


std::string dataPath = "../test/data.bin";
std::string outputPath = "../out/data.out";


int main(int argc, char const* argv[])
{
    if (argc > 4) {
        printf("Invalid Usages.\n");
        printf("Usage: ./main [parallel(0/1)] [cached log(0/1)] [verbose(0/1)]\n");
        exit(-1);
    }
    const int printSample = argc == 4 ? atoi(argv[3]) : 1;
    const int printResult = argc == 4 ? atoi(argv[3]) : 1;
    const int PARALLEL = argc > 1 ? atoi(argv[1]) : 0;
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

    // Read in and process the samples one-by-one
    int ROWS, COLS;
    long t1, t2;
    double time_in_total = 0.0;

    float* input;
    float* output;
    float* logres;

    logres = (float*)malloc(26 * sizeof(float));
    for (int i = 1; i <= 25; i++) {
        logres[i] = log(i);
    }

    while (getNextSample(instream, &COLS, &ROWS, &input) != 0) {

        output = (float*)malloc(ROWS * COLS * sizeof(float));
        if (printSample) {
            // Print out a small portion of the sample
            printf("\nsample: [%d x %d]\n", ROWS, COLS);
            for (int j = ROWS - 5; j < ROWS; j++) {
                for (int i = COLS - 5; i < COLS; i++) {
                    printf("%8.5f ", input[j * COLS + i]);
                }
                printf("\n");
            }
        }

        // getchar();

        t1 = getTime();
        calculateEntropy(input, output, logres, ROWS, COLS, PARALLEL, CACHED_LOG);
        t2 = getTime();


        if (printResult) {
            // Print out a small portion of the result
            printf("result:\n");
            for (int j = ROWS - 5; j < ROWS; j++) {
                for (int i = COLS - 5; i < COLS; i++) {
                    printf("%8.5f ", output[j * COLS + i]);
                }
                printf("\n");
            }
        }

        time_in_total += double(t2 - t1) / 1000000000L;
        printf("Calculation is done in %f second(s).\n", double(t2 - t1) / 1000000000L);


        // Write the result to the output stream
        char buffer[128];
        sprintf(buffer, "%d,", COLS);
        W_CHK(fputs(buffer, outStream));
        sprintf(buffer, "%d,", ROWS);
        W_CHK(fputs(buffer, outStream));
        for (int i = 0; i < ROWS * COLS; i++) {
            sprintf(buffer, "%.5f,", output[i]);
            W_CHK(fputs(buffer, outStream));
        }
        W_CHK(fputs("\n", outStream));


        // De-allocate the sample and the result
        free(input);
        free(output);
        input = NULL;
        output = NULL;
    }

    printf("\n>>> Samples were all done in %f second(s).\n", time_in_total);
    printf(">>> OpenMP Status: %s\n", PARALLEL ? "ON" : "OFF");
    printf(">>> Cached Log(0~25): %s\n", CACHED_LOG ? "ON" : "OFF");

    free(logres);
    logres = NULL;

    // Close the output stream
    fclose(instream);
    fclose(outStream);
    return 0;
}
