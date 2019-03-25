#ifndef CUDA_MACROS_H
#define CUDA_MACROS_H

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
	int maxActiveBlocks;\
\
  	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, func_name##_kernel, 0, 0);\
  	realGridSize = (arr_len + blockSize - 1) / blockSize;\
\
	cudaDeviceProp prop;\
  	cudaGetDeviceProperties(&prop, 0);\
	uint32_t smCount = prop.multiProcessorCount;\
	cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func_name##_kernel, blockSize, 0);\
    if (error == cudaSuccess)\
    	realGridSize = maxActiveBlocks * smCount;\
\
	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;\
	func_name##_kernel<<<realGridSize, blockSize>>>(a_arr, b_arr, c_arr, arr_len);\
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
	int blockSize;\
  	int minGridSize;\
  	int realGridSize;\
\
  	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, func_name##_kernel, 0, 0);\
  	realGridSize = (arr_len + blockSize - 1) / blockSize;\
\
	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;\
\
	func_name##_kernel<<<realGridSize, blockSize>>>(a_arr, b_arr, c_arr, arr_len);\
}

#define GENERAL_TEST_1_ARG_1_TYPE(func_name, type) GENERAL_TEST_1_ARG_2_TYPES(func_name, type, type)

#endif
