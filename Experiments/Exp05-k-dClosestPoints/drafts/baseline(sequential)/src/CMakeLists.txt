cmake_minimum_required(VERSION 3.10)

find_package(CUDA QUIET REQUIRED)

file(GLOB SOURCES "*.h" "*.cpp" "*.cu")

cuda_add_executable(main ${SOURCES})
