#include "benchmark.cuh"
#include "basic_arithmetic.cuh"
#include "square_256_to_512.cuh"
#include "mont_mul.cuh"

#define GENERAL_TEST(func_name) \
__global__ void func_name##_kernel(uint256_g* a_arr, uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(uint256_g* a_arr, uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}


#define MUL_TEST(func_name) \
__global__ void func_name##_kernel(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}

#define SQUARE_TEST(func_name) \
__global__ void func_name##_kernel(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}

using mul_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint512_g>;
using general_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint256_g>;

GENERAL_TEST(add_uint256_naive)
GENERAL_TEST(add_uint256_asm)

general_func_vec_t addition_bench = {
    {"naive approach", add_uint256_naive_driver},
	{"asm", add_uint256_asm_driver}
};

GENERAL_TEST(sub_uint256_naive)
GENERAL_TEST(sub_uint256_asm)

general_func_vec_t substraction_bench = {
    {"naive approach", sub_uint256_naive_driver},
	{"asm", sub_uint256_asm_driver}
};

MUL_TEST(mul_uint256_to_512_asm)
MUL_TEST(mul_uint256_to_512_naive)
MUL_TEST(mul_uint256_to_512_asm_with_allocation)
MUL_TEST(mul_uint256_to_512_asm_longregs)

mul_func_vec_t mul_bench = {
    {"naive approach", mul_uint256_to_512_naive_driver},
	{"asm", mul_uint256_to_512_asm_driver},
	{"asm with register alloc", mul_uint256_to_512_asm_with_allocation_driver},
	{"asm with longregs", mul_uint256_to_512_asm_longregs_driver}
};

SQUARE_TEST(square_uint256_to_512_naive)
SQUARE_TEST(square_uint256_to_512_asm)

mul_func_vec_t square_bench = {
    {"naive approach", square_uint256_to_512_naive_driver},
	{"asm", square_uint256_to_512_asm_driver},
};


GENERAL_TEST(mont_mul_256_naive_SOS)
GENERAL_TEST(mont_mul_256_naive_CIOS)
GENERAL_TEST(mont_mul_256_asm_SOS)
GENERAL_TEST(mont_mul_256_asm_CIOS)

general_func_vec_t mont_mul_bench = {
    {"naive SOS", mont_mul_256_naive_SOS_driver},
	{"asm SOS", mont_mul_256_asm_SOS_driver},
	{"naive CIOS", mont_mul_256_naive_CIOS_driver},
	{"asm CIOS", mont_mul_256_asm_CIOS_driver}
};



int main(int argc, char* argv[])
{
	size_t bench_len = 0x1000000;
	std::cout << "addition benchmark: " << std::endl;
	gpu_benchmark(addition_bench, bench_len);
	std::cout << "substraction benchmark: " << std::endl;
	gpu_benchmark(substraction_bench, bench_len);
	std::cout << "multiplication benchmark: " << std::endl;
	gpu_benchmark(mul_bench, bench_len);
	std::cout << "square benchmark: " << std::endl;
	gpu_benchmark(square_bench, bench_len);
	std::cout << "montgomery multiplication benchmark: " << std::endl;
	gpu_benchmark(mont_mul_bench, bench_len);
    return 0;
}
