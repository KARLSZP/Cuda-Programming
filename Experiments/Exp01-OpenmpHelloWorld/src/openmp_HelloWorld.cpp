/*##########################################
##  姓名：17341137 宋震鹏
##  文件说明：Hello world
##########################################*/
#include <iostream>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <omp.h>
using namespace std;


/*##########################################
##  函数：PrintOpenmpInfo
##  函数描述：打印OpenMP的版本信息
##########################################*/
void PrintOpenmpInfo()
{
    unordered_map<unsigned, string> map {
        {200505, "2.5"}, {200805, "3.0"},
        {201107, "3.1"}, {201307, "4.0"},
        {201511, "4.5"}
    };
    cout << "OpenMP version: " <<  map.at(_OPENMP).c_str() << "." << endl;
}

/*##########################################
##  函数：PrintHello
##  函数描述：并行打印Hello World
##########################################*/
void PrintHello()
{
    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        int max_threads = omp_get_max_threads();
        printf("Hello World (Thread %d of %d)\n", thread, max_threads);
    }
}

int main(void)
{
    PrintOpenmpInfo();
    PrintHello();
    return 0;
}