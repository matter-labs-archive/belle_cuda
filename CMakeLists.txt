cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test_cuda LANGUAGES CXX CUDA)

#string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")


SET(EXECUTABLE_OUTPUT_PATH build)

include_directories(include)
add_subdirectory(sources)