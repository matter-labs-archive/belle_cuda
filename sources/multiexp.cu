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


//1) using warp level reduction and atomics
//---------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_warp_level_atomics(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len, int* mutex)
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
        while (atomicCAS(mutex, 0, 1) != 0);
          
        *out = ECC_ADD(*out, acc);
       
        atomicExch(mutex, 0);
    }  
}

void naive_multiexp_kernel_warp_level_atomics_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
  	int minGridSize;
  	int realGridSize;

    int* mutex;
    cudaMalloc((void**)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_warp_level_atomics, 0, 0);
    realGridSize = (arr_len + blockSize - 1) / blockSize;

	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;

    //create point at infty and copy it to output arr

    ec_point point_at_infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };

    cudaMemcpy(out, &point_at_infty, sizeof(ec_point), cudaMemcpyHostToDevice);

	naive_multiexp_kernel_warp_level_atomics<<<realGridSize, blockSize>>>(point_arr, power_arr, out, arr_len, mutex);

    cudaFree(mutex);
}

//2) using block level reduction and atomics
//---------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_block_level_atomics(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len, int* mutex)
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
        while (atomicCAS(mutex, 0, 1) != 0);
        *out = ECC_ADD(*out, acc);
        atomicExch(mutex, 0);  
    }
}

void naive_multiexp_kernel_block_level_atomics_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
    int minGridSize;
  	int realGridSize;

    int* mutex;
    cudaMalloc((void**)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_atomics, 0, 4 * N * 3 * WARP_SIZE);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;

    ec_point point_at_infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };

    cudaMemcpy(out, &point_at_infty, sizeof(ec_point), cudaMemcpyHostToDevice);

	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_block_level_atomics<<<realGridSize, blockSize>>>(point_arr, power_arr, out, arr_len, mutex);

    cudaFree(mutex);
}

//3) using block level reduction and recursion
//---------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_block_level_recursion(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
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

__global__ void naive_kernel_block_level_reduction(ec_point* in_arr, ec_point* out_arr, size_t arr_len)
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

void naive_multiexp_kernel_block_level_recursion_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_recursion, 0, 4 * N * 3 * WARP_SIZE);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;;
    realGridSize = min(realGridSize, DEFAUL_NUM_OF_THREADS_PER_BLOCK);
    
	std::cout << "Real grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_block_level_recursion<<<realGridSize, blockSize>>>(point_arr, power_arr, out_arr, arr_len);
    cudaDeviceSynchronize();
    naive_kernel_block_level_reduction<<<1, DEFAUL_NUM_OF_THREADS_PER_BLOCK>>>(out_arr, out_arr, realGridSize);
}

//Pippenger
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------

//Pippenger final exponentiation

__global__ void Pippenger_final_exponentiation(ec_point* in_arr, ec_point* out_arr, size_t arr_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
    {   
        ec_point pt = in_arr[tid];

        for (size_t j = 0; j < threadIdx.x; j++)
            pt = ECC_DOUBLE(pt);
        
        out_arr[tid] = pt;

        tid += blockDim.x * gridDim.x;
	}
}

__global__ void multiexp_Pippenger(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len, int* mutex_arr)
{
    ec_point acc = point_at_infty();
    
    size_t start = (arr_len / gridDim.x) * blockIdx.x;
    size_t end = (arr_len / gridDim.x) * (blockIdx.x + 1);

    for (size_t i = start; i < end; i++)
    {
        if (get_bit(power_arr[i], threadIdx.x))
            acc = ECC_MIXED_ADD(acc, point_arr[i]);
    }

    while (atomicCAS(mutex_arr + threadIdx.x, 0, 1) != 0);
    out_arr[threadIdx.x] = ECC_ADD(out_arr[threadIdx.x], acc);
    atomicExch(mutex_arr + threadIdx.x, 0);   
}

void Pippenger_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;

    size_t M = 256;
    int* mutex_arr;
    cudaMalloc((void**)&mutex_arr, sizeof(int) * M);
    cudaMemset(mutex_arr, 0, sizeof(int) * M);

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiexp_Pippenger, 0, 4 * N * 3 * WARP_SIZE);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;;

    //but here we need an array of such elements!

    ec_point point_at_infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };

    for (size_t j = 0 ; j < 256; j++)
    {
        cudaMemcpy(out_arr + j, &point_at_infty, sizeof(ec_point), cudaMemcpyHostToDevice);
    }
    
	std::cout << "Real grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;

	multiexp_Pippenger<<<realGridSize, 256>>>(point_arr, power_arr, out_arr, arr_len, mutex_arr);
    cudaDeviceSynchronize();

    Pippenger_final_exponentiation<<<1, 256>>>(out_arr, out_arr, 256);
    cudaDeviceSynchronize();

    naive_kernel_block_level_reduction<<<1, 256>>>(out_arr, out_arr, 256);

    cudaFree(mutex_arr);
}
