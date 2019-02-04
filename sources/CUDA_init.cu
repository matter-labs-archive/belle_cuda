#include "cuda_structs.h"

#include <stdio.h>

bool CUDA_init()
{
    //first find suitable Cuda device
	//TBD: or split between several CUDA devices if possible
	int device_count;
	cudaError_t cudaStatus = cudaGetDeviceCount(&device_count);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount failed!");
		return false;
	}
	if (device_count == 0)
	{
		fprintf(stderr, "No suitable CUDA devices were found!");
		return false;
	}

	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceProperties(&prop, 0);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaGetDeviceCount failed!");
		return false;
	}

	printf("Compute possibilities: %d.%d\n", prop.major, prop.minor);

	//TODO: check if there are enough constant memory and other additional checks
	//set appropriate device
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return false;
	}

    return true;
}


