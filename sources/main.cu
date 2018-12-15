#include "benchmark.cuh"

__global__ void mul_kernel_naive(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = mul_uint256_to_512_naive(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

void mul_naive_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
	mul_kernel_naive<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);
}

__global__ void mul_kernel_asm(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = mul_uint256_to_512_asm(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

void mul_asm_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
	mul_kernel_asm<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);
}

__global__ void mul_kernel_asm_longregs(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = mul_uint256_to_512_asm_longregs(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

void mul_asm_longregs_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
	mul_kernel_asm_longregs<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);
}

using longmul_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint512_g>;

longmul_func_vec_t bench = {
    {"naive approach", mul_naive_driver},
    {"asm", mul_asm_driver},
    {"longregs asm", mul_asm_longregs_driver}
};

int main(int argc, char* argv[])
{
    gpu_benchmark(bench, 10000000);
    return 0;
}
