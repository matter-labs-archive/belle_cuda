cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test_cuda LANGUAGES CXX CUDA)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61")


SET(EXECUTABLE_OUTPUT_PATH build)

include_directories(include)
add_subdirectory(sources)