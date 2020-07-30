/*#########################################
##  ID: 17341137 ???
##  Introduction:
##     Calculate Entropy - core header
#########################################*/
#include <iostream>
#include <cstdio>
#include <cmath>

using namespace std;

#define WINDOW_SIZE 5
#define MIN_ELEM 0
#define MAX_ELEM 15

// MACRO to check cuda error
#define CHECK(res) { if(res != cudaSuccess){printf("Error %s:%d , ", __FILE__,__LINE__);\
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}


void calculateEntropy(float* input, float* output, float* logres, int rows, int cols, bool SHARED_MEM, bool CACHED_LOG);