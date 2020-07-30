#include <iostream>
#include <cstdio>
#include <cmath>
#include <omp.h>

using namespace std;

#define WINDOW_SIZE 5
#define MIN_ELEM 0
#define MAX_ELEM 15


void calculateEntropy(float* input, float* output, float* logres, int rows, int cols, bool PARALLEL, bool CACHED_LOG);