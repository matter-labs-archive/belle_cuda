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
//3) using warp level reduction and recursion
//4) using block level reduction and recursion

//TODO: it seems that the best way is to combine these approaches, e.g. do several levels of atomic add, than block reduce - there is a vast field
//for experiements

//TODO: implement version with cooperative groups

//we have implemented vectorized loads inspired by: https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

DEVICE_FUNC inline void __shfl_down(const ec_point& in_var, ec_point& out_var, unsigned int offset, int width=32)
{
    //ec_point = 3 * 8  = 24 int4
    const int4* a = reinterpret_cast<const int4*>(&in_var);
    int4* b = reinterpret_cast<int4*>(&out_var);

    for (unsigned i = 0; i < N; i++)
    {
        b[i] = __shfl_down(a[i], offset, width);
    }
}

DEVICE_FUNC inline ec_point warpReduce(ec_point val)
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
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    //Final reduce within first warp
    if (wid==0)
        val = warpReduceSum(val); 

    return val;
}

__global__ void naive_multiexp_kernel_warp_level_atomics(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x
	}

    acc = warpReduceSum(acc);
    
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(out, acc);  
}

void naive_multiexp_kernel_warp_level_atomics_driver(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
  	int minGridSize;
  	int gridSize;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_warp_level_atomics, 0, 0);
  	gridSize = (arr_len + blockSize - 1) / blockSize;

	std::cout << "Grid size: " << gridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_warp_level_atomics<<<gridSize, blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

__global__ void naive_multiexp_kernel_block_level_atomics(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x
	}

    acc = blockReduceSum(acc);
    if (threadIdx.x == 0)
        atomicAdd(out, acc);
}

void naive_multiexp_kernel_block_level_atomics_driver(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
  	int minGridSize;
  	int gridSize;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_atomics, 0, 4 * N * 3 * WARP_SIZE);
  	gridSize = (arr_len + blockSize - 1) / blockSize;

	std::cout << "Grid size: " << gridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_block_level_atomics<<<gridSize, blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

//NB: here arr is the whole array!

__global__ void naive_multiexp_kernel_warp_level_recursion(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x
	}

    acc = warpReduceSum(acc);

  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

__global__ void naive_multiexp_kernel_warp_level_recursion(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
    out[blockIdx.x]=sum;
}

//Pipenger: vasic version - simple, yet powerful. The same version of Pippenger algorithm is implemented in libff and Bellman
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------


