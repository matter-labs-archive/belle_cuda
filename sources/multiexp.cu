#include "cuda_structs.h"

#include <iostream>

//Various algorithms for simultaneous multiexponentiation: naive approach and Pippenger algorithm
//naive approach was widely inspired by https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------

//There are four versions using naive approach:
//1) using warp level reduction and atomics
//2) using block level reduction and atomics
//3) using block level reduction and recursion

//TODO: it seems that the best way is to combine these approaches, e.g. do several levels of atomic add, than block reduce - there is a vast field
//for experiements

//TODO: implement using warp level reduction and recursion

//TODO: implement version with cooperative groups

//TODO: implement approach using CUB library: http://nvlabs.github.io/cub/index.html

//we have implemented vectorized loads inspired by: https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

//Useful miscellaneous functions
//-----------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC inline void __shfl_down(const ec_point& in_var, ec_point& out_var, unsigned int offset, int width=32)
{
    //ec_point = 3 * 8  = 24 int = 6 int4
    const int4* a = reinterpret_cast<const int4*>(&in_var);
    int4* b = reinterpret_cast<int4*>(&out_var);

    for (unsigned i = 0; i < 6; i++)
    {
        b[i].x = __shfl_down_sync(a[i].x, offset, width);
        b[i].y = __shfl_down_sync(a[i].y, offset, width);
        b[i].z = __shfl_down_sync(a[i].z, offset, width);
        b[i].w = __shfl_down_sync(a[i].w, offset, width);
    }
}

DEVICE_FUNC inline ec_point warpReduceSum(ec_point val)
{
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    { 
        ec_point temp;
        __shfl_down(val, temp, offset);
        val = ECC_ADD(val, temp);
    }

    return val;
}

DEVICE_FUNC inline ec_point blockReduceSum(ec_point val)
{
    // Shared mem for 32 partial sums
    static __shared__ ec_point shared[WARP_SIZE]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Each warp performs partial reduction
    val = warpReduceSum(val);     

    // Write reduced value to shared memory
    if (lane==0)
        shared[wid]=val; 

    // Wait for all partial reductions
    __syncthreads();              

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : point_at_infty();

    //Final reduce within first warp
    if (wid == 0)
        val = warpReduceSum(val); 

    return val;
}

struct Lock
{
    int* mutex;

    Lock()
    {
        int state = 0;
        cudaMalloc((void**)&mutex, sizeof(int));
        cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
    }

    ~Lock()
    {
        cudaFree(mutex);
    }

    DEVICE_FUNC void lock()
    {
        while (atomicCAS(mutex, 0, 1) != 0);
    }

    DEVICE_FUNC void unlock()
    {
        atomicExch(mutex, 0);
    }
};


//1) using warp level reduction and atomics
//---------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_warp_level_atomics(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len, Lock lock)
{
	ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x;
	}

    acc = warpReduceSum(acc);
    
    if ((threadIdx.x & (warpSize - 1)) == 0)
    {
        lock.lock();
        *out = ECC_ADD(*out, acc);
        lock.unlock();
    }  
}

void naive_multiexp_kernel_warp_level_atomics_driver(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
  	int minGridSize;
  	int realGridSize;
	int optimalGridSize;

    Lock lock;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_warp_level_atomics, 0, 0);
    realGridSize = (arr_len + blockSize - 1) / blockSize;
	optimalGridSize = min(minGridSize, realGridSize);

	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_warp_level_atomics<<<optimalGridSize, blockSize>>>(point_arr, power_arr, out, arr_len, lock);
}

//2) using block level reduction and atomics
//---------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_block_level_atomics(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len, Lock lock)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x;
	}

    acc = blockReduceSum(acc);
    if (threadIdx.x == 0)
    {
        lock.lock();
        *out = ECC_ADD(*out, acc);
        lock.unlock();
    }
}

void naive_multiexp_kernel_block_level_atomics_driver(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
    int minGridSize;
  	int realGridSize;
	int optimalGridSize;

    Lock lock;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_atomics, 0, 4 * N * 3 * WARP_SIZE);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;
	optimalGridSize = min(minGridSize, realGridSize);

	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_block_level_atomics<<<optimalGridSize, blockSize>>>(point_arr, power_arr, out, arr_len, lock);
}

//4) using block level reduction and recursion
//---------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_block_level_recursion(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x;
	}

    acc = blockReduceSum(acc);
    
    if (threadIdx.x == 0)
        out_arr[blockIdx.x] = acc;
}

__global__ void naive_kernel_block_level_reduction(const ec_point* in_arr, ec_point* out_arr, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
    {   
        acc = ECC_ADD(acc, in_arr[tid]);
        tid += blockDim.x * gridDim.x;
	}

    acc = blockReduceSum(acc);

    if (threadIdx.x == 0)
        out_arr[blockIdx.x] = acc;
}

void naive_multiexp_kernel_block_level_recursion_driver(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;
	int optimalGridSize;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_recursion, 0, 4 * N * 3 * WARP_SIZE);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;
	optimalGridSize = min(minGridSize, realGridSize);
    optimalGridSize = min(optimalGridSize, DEFAUL_NUM_OF_THREADS_PER_BLOCK);
    
	std::cout << "Real grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_block_level_recursion<<<optimalGridSize, blockSize>>>(point_arr, power_arr, out_arr, arr_len);
    naive_kernel_block_level_reduction<<<1, DEFAUL_NUM_OF_THREADS_PER_BLOCK>>>(out_arr, out_arr, optimalGridSize);
}

