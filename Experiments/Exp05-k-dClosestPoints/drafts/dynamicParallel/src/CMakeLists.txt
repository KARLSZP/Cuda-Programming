cmake_minimum_required(VERSION 3.10)

find_package(CUDA QUIET REQUIRED)

file(GLOB SOURCES "*.h" "*.cpp" "*.cu")

set(CUDA_SEPARABLE_COMPILATION TRUE)
list(APPEND CUDA_NVCC_FLAGS -arch=compute_61)

cuda_add_executable(main ${SOURCES})

target_link_libraries(main ${CUDA_cudadevrt_LIBRARY})
