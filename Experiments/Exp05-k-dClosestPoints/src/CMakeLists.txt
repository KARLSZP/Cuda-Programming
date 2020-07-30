cmake_minimum_required(VERSION 2.8)

find_package(CUDA QUIET REQUIRED)

file(GLOB SOURCES "src/*.h" "src/*.cpp" "src/*.cu")
# file(GLOB SOURCES "*.h" "*.cpp" "*.cu")

set(CUDA_SEPARABLE_COMPILATION TRUE)
list(APPEND CUDA_NVCC_FLAGS -arch=compute_61)

cuda_add_executable(main ${SOURCES})

target_link_libraries(main ${CUDA_cudadevrt_LIBRARY})
