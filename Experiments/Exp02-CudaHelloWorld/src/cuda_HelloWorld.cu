/*#########################################
##  姓名：17341137 宋震鹏
##  文件说明：Hello World on Cuda
#########################################*/
#include "cuda_HelloWorld.cuh"

using namespace std;

#define CHECK(res) { if(res != cudaSuccess){printf("Error ：%s:%d , ", __FILE__,__LINE__);\
printf("code : %d , reason : %s \n", res,cudaGetErrorString(res));exit(-1);}}

/*#########################################
##  函数：CheckDevice
##  函数描述：打印设备基本信息
##########################################*/
void CheckDevice()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        double PMB = 2.0 * prop.memoryClockRate * ((prop.memoryBusWidth / 8) / 1.0e6);
        printf("Device Number: %d\n", i);
        printf("Device name  : %s\n", prop.name);
        printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("Peak Memory Bandwidth (GB/s): %f\n", PMB);
    }
}

__global__ void SHelloWorld()
{
    printf("SubHelloWorld from %d-%d\n", blockIdx.x, threadIdx.x);
}

__global__ void HelloWorld()
{
    printf("HelloWorld from %d-%d\n", blockIdx.x, threadIdx.x);
    SHelloWorld <<< 2, 128>>>();
}

void SayHello()
{
    HelloWorld <<< 4, 8>>>();
    CHECK(cudaDeviceSynchronize());
}

// int main()
// {
//     CheckDevice();
//     SayHello();
//     return 0;
// }