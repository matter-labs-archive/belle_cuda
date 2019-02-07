#include "cuda_structs.h"
#include <iostream>

#define GENERAL_TEST_2_ARGS_3_TYPES(func_name, A_type, B_type, C_type) \
__global__ void func_name##_kernel(A_type *a_arr, B_type *b_arr, C_type *c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(A_type *a_arr, B_type *b_arr, C_type *c_arr, size_t arr_len)\
{\
	int blockSize;\
  	int minGridSize;\
  	int realGridSize;\
	int optimalGridSize;\
\
  	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, func_name##_kernel, 0, 0);\
  	realGridSize = (arr_len + blockSize - 1) / blockSize;\
	optimalGridSize = min(minGridSize, realGridSize);\
\
	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;\
	func_name##_kernel<<<optimalGridSize, blockSize>>>(a_arr, b_arr, c_arr, arr_len);\
}

#define GENERAL_TEST_2_ARGS_2_TYPES(func_name, input_type, output_type) GENERAL_TEST_2_ARGS_3_TYPES(func_name, input_type, input_type, output_type)
#define GENERAL_TEST_2_ARGS_1_TYPE(func_name, type) GENERAL_TEST_2_ARGS_3_TYPES(func_name, type, type, type)

#define GENERAL_TEST_1_ARG_2_TYPES(func_name, input_type, output_type) \
__global__ void func_name##_kernel(input_type *a_arr, input_type *b_arr, output_type *c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(input_type *a_arr, input_type *b_arr, output_type *c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}

#define GENERAL_TEST_1_ARG_1_TYPE(func_name, type) GENERAL_TEST_1_ARG_2_TYPES(func_name, type, type)


GENERAL_TEST_2_ARGS_1_TYPE(add_uint256_naive, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(add_uint256_asm, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(sub_uint256_naive, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(sub_uint256_asm, uint256_g)

GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_naive, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm_with_allocation, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm_longregs, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_Karatsuba, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm_with_shuffle, uint256_g, uint512_g)


GENERAL_TEST_1_ARG_2_TYPES(square_uint256_to_512_naive, uint256_g, uint512_g)
GENERAL_TEST_1_ARG_2_TYPES(square_uint256_to_512_asm, uint256_g, uint512_g)

GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_naive_SOS, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_naive_CIOS, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_asm_SOS, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_asm_CIOS, uint256_g)

GENERAL_TEST_2_ARGS_1_TYPE(ECC_ADD_PROJ, ec_point);
GENERAL_TEST_2_ARGS_1_TYPE(ECC_SUB_PROJ, ec_point);
GENERAL_TEST_1_ARG_1_TYPE(ECC_DOUBLE_PROJ, ec_point);
GENERAL_TEST_1_ARG_2_TYPES(IS_ON_CURVE_PROJ, ec_point, bool);

GENERAL_TEST_2_ARGS_1_TYPE(ECC_ADD_JAC, ec_point);
GENERAL_TEST_2_ARGS_1_TYPE(ECC_SUB_JAC, ec_point);
GENERAL_TEST_1_ARG_1_TYPE(ECC_DOUBLE_JAC, ec_point);
GENERAL_TEST_1_ARG_2_TYPES(IS_ON_CURVE_JAC, ec_point, bool);

GENERAL_TEST_2_ARGS_3_TYPES(ECC_double_and_add_exp_PROJ, ec_point, uint256_g, ec_point);
GENERAL_TEST_2_ARGS_3_TYPES(ECC_ternary_expansion_exp_PROJ, ec_point, uint256_g, ec_point);
GENERAL_TEST_2_ARGS_3_TYPES(ECC_double_and_add_exp_JAC, ec_point, uint256_g, ec_point);
GENERAL_TEST_2_ARGS_3_TYPES(ECC_ternary_expansion_exp_JAC, ec_point, uint256_g, ec_point);