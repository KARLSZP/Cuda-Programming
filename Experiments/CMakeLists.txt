# project(cuda_helloworld)

cmake_minimum_required(VERSION 3.10)
find_package(CUDA QUIET REQUIRED)

# add_subdirectory(cuda) set(EXTRA_LIBS ${EXTRA_LIBS} gpu)

# add_executable(cuda_helloworld main.cpp)

# target_link_libraries(cuda_helloworld ${EXTRA_LIBS})

cuda_add_executable(main main.cu cuda/cuda_HelloWorld.cu)
