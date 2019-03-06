#include "cuda_structs.h"


//Sime useful links:

//What about dealing with tail effects?
//https://devblogs.nvidia.com/cuda-pro-tip-minimize-the-tail-effect/

//efficient finite field library:
//http://mpfq.gforge.inria.fr/doc/index.html

//additional tricks for synchronization:
//https://habr.com/ru/post/151897/

//Why we do coalesce memory accesses:
//https://devblogs.nvidia.com/how-access-global-memory-efficiently-cuda-c-kernels/

//check strided kernels: look at XMP as a source of inspiration
//TODO: investigate why it works

#define GEOMETRY 128

__global__ void xmpC2S_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out)
{
  //outer dimension = N
  //inner dimension = limbs
  
  //read strided in inner dimension`
  //write coalesced in outer dimension
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    for(uint32_t j=blockIdx.y*blockDim.y+threadIdx.y;j<limbs;j+=blockDim.y*gridDim.y) {
      out[j*stride + i] = in[i*limbs + j];
    }
  }
}

inline void xmpC2S(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out, cudaStream_t stream) {
  dim3 threads, blocks;

  //target 128 threads
  threads.x=MIN(32,N);
  threads.y=MIN(DIV_ROUND_UP(128,threads.x),limbs);

  blocks.x=DIV_ROUND_UP(N,threads.x);
  blocks.y=DIV_ROUND_UP(limbs,threads.y);

  //convert from climbs to slimbs
  xmpC2S_kernel<<<blocks,threads,0,stream>>>(N,limbs,stride,in,out);
}

inline void xmpS2C(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out, cudaStream_t stream) {
  dim3 threads, blocks;

  //target 128 threads
  threads.x=MIN(32,limbs);
  threads.y=MIN(DIV_ROUND_UP(128,threads.x),N);

  blocks.x=DIV_ROUND_UP(limbs,threads.x);
  blocks.y=DIV_ROUND_UP(N,threads.y);

  //convert from climbs to slimbs
  xmpS2C_kernel<<<blocks,threads,0,stream>>>(N,limbs,stride,in,out);
}

//x along limbs
//y along N
__global__ void xmpS2C_kernel(uint32_t N, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out) {
  //outer dimension = limbs
  //inner dimension = N

  //read strided in inner dimension
  //write coalesced in outer dimension
  for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;i<limbs;i+=blockDim.x*gridDim.x) {
    for(uint32_t j=blockIdx.y*blockDim.y+threadIdx.y;j<N;j+=blockDim.y*gridDim.y) {
      out[j*limbs + i] = in[i*stride + j];
    }
  }
}


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
    if (error == cudaSuccess && realGridSize > maxActiveBlocks * smCount)\
    	realGridSize = maxActiveBlocks * smCount;\
\
	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;\
	func_name##_kernel<<<realGridSize, blockSize>>>(a_arr, b_arr, c_arr, arr_len);\
}

//warp-based elliptic point addition

DEVICE_FUNC ec_point ECC_ADD_PROJ(const ec_point& left, const ec_point& right)
{
	if (is_infinity(left))
		return right;
	if (is_infinity(right))
		return left;

	uint256_g U1, U2, V1, V2;

  //1
	U1 = MONT_MUL(left.z, right.y);
  //2
	U2 = MONT_MUL(left.y, right.z);
  //3
	V1 = MONT_MUL(left.z, right.x);
  //4
	V2 = MONT_MUL(left.x, right.z);

	ec_point res;

	if (EQUAL(V1, V2))
	{
		if (!EQUAL(U1, U2))
			return point_at_infty();
		else
			return  ECC_DOUBLE_PROJ(left);
	}

  //Transmit phase: 2 -> 1, 4 -> 3

  //1
	uint256_g U = FIELD_SUB(U1, U2);
  //2
	uint256_g V = FIELD_SUB(V1, V2);
  //3
	uint256_g W = MONT_MUL(left.z, right.z);
	uint256_g Vsq = MONT_SQUARE(V);
	uint256_g Vcube = MONT_MUL(Vsq, V);

	uint256_g temp1, temp2;
	temp1 = MONT_SQUARE(U);
	temp1 = MONT_MUL(temp1, W);
	temp1 = FIELD_SUB(temp1, Vcube);
	temp2 = MONT_MUL(BASE_FIELD_R2, Vsq);
	temp2 = MONT_MUL(temp2, V2);
	uint256_g A = FIELD_SUB(temp1, temp2);
	res.x = MONT_MUL(V, A);

	temp1 = MONT_MUL(Vsq, V2);
	temp1 = FIELD_SUB(temp1, A);
	temp1 = MONT_MUL(U, temp1);
	temp2 = MONT_MUL(Vcube, U2);
	res.y = FIELD_SUB(temp1, temp2);

	res.z = MONT_MUL(Vcube, W);
	return res;
}

//how to reduce:
//DEVICE_FUNC uint256_g mont_mul_256_asm_CIOS(const uint256_g& u, const uint256_g& v)