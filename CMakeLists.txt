cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test_cuda LANGUAGES CXX CUDA)

#Determine compute possibilities of device installed on target system

set(OUTPUTFILE ${CMAKE_BINARY_DIR}/check_CUDA_arch) 
set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/utilities/check_CUDA_arch.cu)
execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE GPU_ARCH)

if(${CUDA_RETURN_CODE} EQUAL 0)
    message(STATUS "GPU Architecture: ${GPU_ARCH}")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}")
else()
    message( FATAL_ERROR ${GPU_ARCH})
endif()



include_directories(include)
add_subdirectory(sources)