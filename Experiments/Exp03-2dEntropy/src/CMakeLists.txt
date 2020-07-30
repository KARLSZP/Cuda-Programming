# project(cuda_helloworld)

cmake_minimum_required(VERSION 3.10)
find_package(CUDA QUIET REQUIRED)

# add_subdirectory(cuda) set(EXTRA_LIBS ${EXTRA_LIBS} gpu)

# add_executable(cuda_helloworld main.cpp)

# target_link_libraries(cuda_helloworld ${EXTRA_LIBS})

file(
  GLOB
  SOURCES
  "cuda/*.cuh"
  "cuda/*.cu"
  "utils/*.h"
  "utils/*.cpp"
  "utils/*.hpp"
  "utils/*.c"
  "*.cpp"
  "*.hpp"
  "*.c"
  "*.h"
  "*.cu")

cuda_add_executable(main ${SOURCES})
