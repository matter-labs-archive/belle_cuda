#include "cuda_structs.h"

#include <iostream>

void get_device_info()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

	uint32_t sm_count = prop.multiProcessorCount;
    uint32_t warp_size = prop.warpSize;
	uint32_t shared_mem_per_block = prop.sharedMemPerBlock;
    uint32_t shared_mem_per_multiprocessor = prop.sharedMemPerMultiprocessor;
	uint32_t regs_per_block = prop.regsPerBlock;
    uint32_t regs_per_multiprocessor = prop.regsPerMultiprocessor;
	uint32_t max_threads_per_block = prop.maxThreadsPerBlock;
	uint32_t max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;

    std::cout << "SM count: " << sm_count << std::endl;
    std::cout << "warp size: " << warp_size << std::endl;
    std::cout << "Number of shared memory per block (in bytes): " << shared_mem_per_block << std::endl;
    std::cout << "Number of shared memory per multiprocessor (in bytes): " << shared_mem_per_multiprocessor << std::endl;
    std::cout << "Number of 32bit registers per block: " << regs_per_block << std::endl;
    std::cout << "Number of 32bit register per multiprocessor: " << regs_per_multiprocessor << std::endl;
    std::cout << "Max number of threads per block: " << max_threads_per_block << std::endl;
    std::cout << "Max number of threads per multiprocessor: " << max_threads_per_multiprocessor << std::endl;   
}