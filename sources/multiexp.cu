#include "cuda_structs.h"

//Various algorithms for simultaneous multiexponentiation: naive approach and Pippenger algorithm
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------

//naive version

__global__ void naive_multiexp_kernel(affine_point* point_arr, uint256_g* power_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
