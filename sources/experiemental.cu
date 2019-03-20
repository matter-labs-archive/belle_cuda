#include "cuda_structs.h"

//how to reduce:
//DEVICE_FUNC uint256_g mont_mul_256_asm_CIOS(const uint256_g& u, const uint256_g& v)

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

//-----------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------

//strided version of multiplication

#define GEOMETRY 128
#define ROUND_UP(n,d) (((n)+(d)-1)/(d)*(d))
#define DIV_ROUND_UP(n,d) (((n)+(d)-1)/(d))

struct geometry2
{
    int gridSize;
    int blockSize;
};

template<typename T>
geometry2 find_geometry2(T func, uint shared_memory_used, uint32_t smCount)
{
    int gridSize;
    int blockSize;
    int maxActiveBlocks;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, shared_memory_used, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, shared_memory_used);
    gridSize = maxActiveBlocks * smCount;

    return geometry2{gridSize, blockSize};
}

__global__ void xmpC2S_kernel(uint32_t count, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out)
{
    //outer dimension = count
    //inner dimension = limbs

    //read strided in inner dimension`
    //write coalesced in outer dimension
    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x*gridDim.x)
    {
        for(uint32_t j=blockIdx.y * blockDim.y + threadIdx.y; j < limbs; j += blockDim.y * gridDim.y)
        {
            out[j*stride + i] = in[i*limbs + j];
        }
    }
}

inline void xmpC2S(uint32_t count, const uint32_t * in, uint32_t * out)
{
    dim3 threads, blocks;
    uint32_t limbs = N;
    //round up to 128 bше boundarн
    uint32_t stride = ROUND_UP(count, 32);  

    //target 128 threads
    threads.x = MIN(32, count);
    threads.y = MIN(DIV_ROUND_UP(128, threads.x), limbs);

    blocks.x = DIV_ROUND_UP(count, threads.x);
    blocks.y=DIV_ROUND_UP(limbs, threads.y);

    //convert from climbs to slimbs
    xmpC2S_kernel<<<blocks,threads>>>(count, limbs, stride, in, out);
}

__global__ void xmpS2C_kernel(uint32_t count, uint32_t limbs, uint32_t stride, const uint32_t * in, uint32_t * out)
{
    //outer dimension = limbs
    //inner dimension = N

    //read strided in inner dimension
    //write coalesced in outer dimension
    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < limbs; i += blockDim.x * gridDim.x)
    {
        for(uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < count; j + =blockDim.y * gridDim.y)
        {
            out[j * limbs + i] = in[ i * stride + j];
        }
    }
}

inline void xmpS2C(uint32_t count, const uint32_t * in, uint32_t * out)
{
    dim3 threads, blocks;
    uint32_t limbs = N;
    //round up to 128 bше boundarн

    //target 128 threads
    threads.x = MIN(32, limbs);
    threads.y = MIN(DIV_ROUND_UP(128, threads.x), count);

    blocks.x=DIV_ROUND_UP(count, threads.x);
    blocks.y=DIV_ROUND_UP(limbs, threads.y);

    //convert from climbs to slimbs
    xmpS2C_kernel<<<blocks,threads>>>(count, limbs, stride, in, out);
}

#define STRIDED_MONT_MUL_TEST(func_name) \
__global__ void func_name##_kernel_strided(uint32_t* a_arr, uint32_t* b_arr, uint32_t* c_arr, size_t count)\
{\
    uint32_t stride = ROUND_UP(count, 32);\
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
    uint256_g A, B, C;\
    while (tid < count)\
    {\
        uint32_t* a_data = a_arr + tid;\
        uint32_t* b_data = b_arr + tid;\
        uint32_t* c_data = c_arr + tid;\
        #pragma unroll\
        for(uint32_t index = 0; index < N; index++)\
        {\
            A.n[index] = a_data[index * stride];\
            B.n[index] = b_data[index * stride];\
        }\
\
        C = func_name(A, B);\
\
        #pragma unroll\
        for(uint32_t index = 0; index < N; index++)\
        {\
            c_data[index * stride]= C[index];
        }\
    }\
}\
\
void func_name##_driver_strided(uint256_g* a_arr, uint256_g* b_arr, uin256_g* c_arr, size_t count)\
{\       
    cudaDeviceProp prop;\
    cudaGetDeviceProperties(&prop, 0);\
    uint32_t smCount = prop.multiProcessorCount;\   
    geometry2 geometry = find_geometry2(T func, 0, uint32_t smCount);\
\
    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl;\
    func_name##_kernel<<<geometry.gridSize, geometry.blockSize>>>(reinterpet_cast<uint32_t*>(a_arr), reinterpet_cast<uint32_t*>(b_arr),\ 
        reinterpet_cast<uint32_t*>(c_arr), count);\
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------

//warp-based long multiplication and montgomery multiplication

#define THREADS_PER_MUL 8

#define UNROLLED_CYCLE_ITER(idx) \
"shfl.sync.idx.b32  x, b, idx, 8;" \
"mad.lo.cc.u32 v, a, x, v;\n\t" \
"madc.hi.cc.u32 u, a, x, u;\n\t" \
"addc.u32 t 0, 0;\n\t" \
"shfl.sync.up.b32  v, v, 1, 8;" \
"shfl.sync.up.b32  c, c, 1, 8;" \
"and.type y, %laneid, 7;\n\t" \
"setp.eq.u32  p, y, 7;\n\t"
"@p {\n\t" \
"mov.u32 c v;\n\t" \
"mov.u32 v 0;\n\t"
"}\n\t" \
"add.cc.u32 v, u, v;\n\t" \
"addc.u32 u, t, 0;\n\t"

//The following implementation is based on paper "A Warp-synchronous Implementation for Multiple-length Multiplication on the GPU"

DEVICE_FUNC uint64_t asm_mul_warp_based(uint32_t A, const uint32_t B)
{
    uint64_t res;
    
    asm(    "{\n\t"  
            ".reg .u32 a, b, x, y, u, v, c, t;\n\t"
            ".reg .pred p;\n\t"

            // "ld.global.u32 a, [A + %laneid];\n\t"
            // "ld.global.u32 b, [B + %laneid];\n\t"
            "mov.b32 a, %0;\n\t"
            "mov.b32 b, %1;\n\t"

            "mov.u32 u, 0;\n\t"
            "mov.u32 v, 0;\n\t"
            "mov.u32 c, 0;\n\t"

            UNROLLED_CYCLE_ITER(0)
            UNROLLED_CYCLE_ITER(1)
            UNROLLED_CYCLE_ITER(2)
            UNROLLED_CYCLE_ITER(3)
            UNROLLED_CYCLE_ITER(4)
            UNROLLED_CYCLE_ITER(5)
            UNROLLED_CYCLE_ITER(6)
            UNROLLED_CYCLE_ITER(7)

            "L1:\n\t"
            "setp.eq.u32   p, ne, 0;\n\t"
            "vote.sync.any.pred  p, p, 8;\n\t"
            "@!p bra L2;\n\t"
            "shfl.sync.down.b32  u, u, 1, 8;\n\t"
            "add.cc.u32 v, v, u;\n\t"
            "addc.u32 u, 0, 0;\n\t"
            "bra L1;\n\t"
            
            "L2:\n\t"

            // "st.global.u32 [OUT + %laneid], c;\n\t"
            // "st.global.u32 [OUT + %laneid + 8], v;\n\t"
            "mov.b64 %2, {c, v};}\n\t"  
            : "=l"(res) : "r"(A), "r"(B));
    
    return res;
}

//I bet there must be a faster solution!
//TODO: have a look at https://ieeexplore.ieee.org/document/5669278

__global__ void warp_based_mul_kernel(const uint256_g* a_arr, const uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idx = tid / THREADS_PER_MUL;
    size_t lane = tid % THREADS_PER_MUL;
	while (idx < arr_len)
	{
		uint32_t A = a_arr[idx].n[lane];
        uint32_t B = b_arr[idx].n[lane];
        uint64_t C = asm_mul_warp_based(A, B, C);

        c_arr[idx].n[lane] = C.low;
        c_arr[idx].n[lane + N] = C.high;

		tid += (blockDim.x * gridDim.x;) / THREADS_PER_MUL;
	}
}

void warp_based_mul_driver(const uint256_g* a_arr, const uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t smCount = prop.multiProcessorCount;   
    geometry2 geometry = find_geometry2(warp_based_mul_kernel, 0, uint32_t smCount);

    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl;
    warp_based_mul_kernel<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

//LARGE REDC Montgomety mul

DEVICE_FUNC __inline__ void long_in_place_add(uint32_t& A, uint32_t B, uint32_t& carry)
{
     asm(   "{\n\t"  
            "add.cc.u32 %0, %0, %2;\n\t"
            "addc.u32 %1, 0, 0;\n\t" 
            : "+r"(A), "=r"(carry) : "r"(B));
}

DEVICE_FUNC __inline__ void long_in_place_sub(uint32_t& A, uint32_t B, uint32_t& carry)
{
     asm(   "{\n\t"  
            "sub.cc.u32 %0, %0, %2;\n\t"
            "addc.u32 %1, 0, 0;\n\t" 
            : "+r"(A), "=r"(carry) : "r"(B));
}

#define MAX_UINT32_VAL 0xffffffff

DEVICE_FUNC __inline__ bool CHECK_CONDITIONS(uint32_t cond1_mask, uint32_t cond2_mask, uint32_t warp_idx)
{
    return ( __clz((cond2_mask << ((3 - warp_idx) * THREADS_PER_MUL)) ^ MAX_UINT32_VAL) >= __clz(cond1_mask << ((3 - warp_idx) * THREADS_PER_MUL)));
}

DEVICE_FUNC void mont_mul_warp_based(const uint256_g& A, const uint256_g& B, uint256_g& OUT)
{
    // T = A * B
    //m = ((T mod R) * N) mod R
    //t = (T + m * p) / R
    //if t >= N then t = t - N
       
    size_t lane = tid % THREADS_PER_MUL;
    size_t warp_idx = tid / THREADS_PER_MUL;
    uint32_t mask = (THREADS_PER_MUL - 1) << (warp_idx * THREADS_PER_MUL);

    uint64_t temp1 = asm_mul_warp_based(A.n[lane], B.n[lane]);
    uint64_t temp2 = asm_mul_warp_based(temp1.low, BASE_FIELD_N_LARGE.n[lane]);
    temp2 = asm_mul_warp_based(temp2, BASE_FIELD_P.n[lane]);

    //adding higher 8-words part
    uint32_t carry = 0;
    long_in_place_add(temp1.high, temp2.high, carry);

    //we are going to check if there is overflow from lower 8-words part
    uint32_t sum = temp1.low + temp2.low;
    uint32_t cond1_mask = __ballot_sync(mask, sum < temp1.low);
    uint32_t cond2_mask = _ballot_sync(mask, sum == MAX_UINT32_VAL);
    if (lane == THREADS_PER_MUL - 1)
        carry = (uint32_t)CHECK_CONDITIONS(cond1, cond2, warp_idx);

    //propagate carry
    while (__any_sync(mask, carry != 0))
    {
        carry = shfl_up_sync(mask, carry, 1, THREADS_PER_MUL);
        long_in_place_add(temp1.high, carry, carry);
        if (lane == THREADS_PER_MUL - 1)
            carry = 0;
    }

    //now temp1.high holds t, compare t with N:
    cond1 = __ballot_sync(mask, temp1.high > BASE_FIELD_P.n[lane]);
    cond2 = __ballot_sync(mask, temp1.high == BASE_FIELD_P.n[lane]);
    if (CHECK_CONDITIONS(cons1, cond2, warp_idx))
    {
        //lane based substraction
        long_in_place_sub(temp1.high, BASE_FIELD_P.n[lane], carry);

        //propagate borrow
        while (__any_sync(mask, carry != 0))
        {
            carry = shfl_up_sync(mask, carry, 1, THREADS_PER_MUL);
            long_in_place_sub(temp1.high, carry, carry);
  
        }
    }

    OUT.n[lane] = temp1.high;
}

__global__ void warp_based_mont_mul_kernel(const uint256_g* a_arr, const uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idx = tid / THREADS_PER_MUL;
	while (idx < arr_len)
	{
		uint256_g& A = a_arr[tid];
        uint256_g& B = b_arr[tid];
        uint256_g& C = c_arr[tid];

        mont_mul_warp_based(A, B, C);
		tid += (blockDim.x * gridDim.x;) / THREADS_PER_MUL;
	}
}

void warp_based_mont_mul_driver(const uint256_g* a_arr, const uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t smCount = prop.multiProcessorCount;   
    geometry2 geometry = find_geometry2(warp_based_mont_mul_kernel, 0, uint32_t smCount);

    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl;
    warp_based_mont_mul_kernel<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------

//warp-based montgomery multiplication and elliptic curve point addition

//16 threads are used to calculate one operation

#define THREADS_PER_ECC_ADD 16

DEVICE_FUNC __inline__ bool is_leader_lane()
{
    return (threadIdx.x % THREADS_PER_ECC_ADD == 0);
}

DEVICE_FUNC __inline__ uint256_g& subwarp_chooser()



DEVICE_FUNC void ECC_add_proj_warp_based(const ec_point* A, const ec_point* B, ec_point* C)
{
    uint32_t exit_flag = 0;

    if (is_leader_lane())
    {
        if (is_infinity(*A))
        {
            *C = *B;
            exit_flag = 1;
        }
	    else if (is_infinity(*B))
        {
		    *C = *A;
            exit_flag = 1;
        }
    }

    exit_flag = __shfl_sync(0xFFFFFFFF, exit_flag, 0, THREADS_PER_ECC_ADD);
    if (exit_flag)
        return;

    uint32_t x, y;

	uint256_g U1, U2, V1, V2;
	U1 = MONT_MUL(left.z, right.y);
	U2 = MONT_MUL(left.y, right.z);
	V1 = MONT_MUL(left.z, right.x);
	V2 = MONT_MUL(left.x, right.z);

	ec_point res;

	if (EQUAL(V1, V2))
	{
		if (!EQUAL(U1, U2))
			return point_at_infty();
		else
			return  ECC_DOUBLE_PROJ(left);
	}

	uint256_g U = FIELD_SUB(U1, U2);
	uint256_g V = FIELD_SUB(V1, V2);
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

__global__ void ECC_add_proj_warp_based_kernel(const ec_point* a_arr, const ec_point* b_arr, ec_point *c_arr, size_t arr_len)
{
	size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) / THREADS_PER_OP;
	while (tid < arr_len)
	{
		ECC_add_proj_warp_based(a_arr + tid, b_arr + tid, c_arr + tid);
		tid += (blockDim.x * gridDim.x) / 16;
	}
}
    
DEVICE_FUNC ec_point ECC_ADD_PROJ(const ec_point& left, const ec_point& right)
{
	if (is_infinity(left))
		return right;
	if (is_infinity(right))
		return left;

	uint256_g U1, U2, V1, V2;
	U1 = MONT_MUL(left.z, right.y);
	U2 = MONT_MUL(left.y, right.z);
	V1 = MONT_MUL(left.z, right.x);
	V2 = MONT_MUL(left.x, right.z);

	ec_point res;

	if (EQUAL(V1, V2))
	{
		if (!EQUAL(U1, U2))
			return point_at_infty();
		else
			return  ECC_DOUBLE_PROJ(left);
	}

	uint256_g U = FIELD_SUB(U1, U2);
	uint256_g V = FIELD_SUB(V1, V2);
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

//Transmit phase: 2 -> 1, 4 -> 3

//1
uint256_g U = FIELD_SUB(U1, U2);
//2
uint256_g V = FIELD_SUB(V1, V2);

uint256_g temp1, temp2;

//1
uint256_g Vsq = MONT_SQUARE(V);
//2
temp1 = MONT_SQUARE(U);
//3
uint256_g W = MONT_MUL(left.z, right.z);

//1
uint256_g Vcube = MONT_MUL(Vsq, V);
//2
temp1 = MONT_MUL(temp1, W);
//3
temp2 = MONT_MUL(BASE_FIELD_R2, Vsq);
//4
tempx = MONT_MUL(Vsq, V2);

//1
temp2 = MONT_MUL(temp2, V2);
//2
temp1 = FIELD_SUB(temp1, Vcube);
//3
tempg = MONT_MUL(Vcube, U2);
//4
res.z = MONT_MUL(Vcube, W);

uint256_g A = FIELD_SUB(temp1, temp2);
res.x = MONT_MUL(V, A);


temp1 = FIELD_SUB(tempx, A);
temp1 = MONT_MUL(U, temp1);

res.y = FIELD_SUB(temp1, tempg);


return res;
}

DEVICE_FUNC ec_point ECC_DOUBLE_PROJ(const ec_point& pt)
{
	if (is_zero(pt.y) || is_infinity(pt))
		return point_at_infty();
	else
	{
		uint256_g temp, temp2;
		uint256_g W, S, B, H, S2;
		ec_point res;

#ifdef BN256_SPECIFIC_OPTIMIZATION
 		temp = MONT_SQUARE(pt.x);
 		W = MONT_MUL(temp, R3_g);
#else
 		temp = MONT_SQUARE(pt.x);
 		temp = MONT_MUL(temp, BASE_FIELD_R3);
 		temp2 = MONT_SQUARE(pt.z);
 		temp2 = MONT_MUL(temp2, CURVE_A_COEFF);
 		W = FIELD_ADD(temp, temp2);
#endif
 		S = MONT_MUL(pt.y, pt.z);
		temp = MONT_MUL(pt.x, pt.y);
 		B = MONT_MUL(temp, S);
		res.x = W;

 		temp = MONT_SQUARE(W);
 		temp2 = MONT_MUL(BASE_FIELD_R8, B);
 		H = FIELD_SUB(temp, temp2);

 		temp = MONT_MUL(BASE_FIELD_R2, H);
 		res.x = MONT_MUL(temp, S);
		
 		//NB: here result is also equal to one of the operands and hence may be reused!!!
 		//NB: this is in fact another possibility for optimization!
 		S2 = MONT_SQUARE(S);
 		temp = MONT_MUL(BASE_FIELD_R4, B);
 		temp = FIELD_SUB(temp, H);
 		temp = MONT_MUL(W, temp);
		
 		temp2 = MONT_SQUARE(pt.y);
 		temp2 = MONT_MUL(BASE_FIELD_R8, temp2);
 		temp2 = MONT_MUL(temp2, S2);
 		res.y = FIELD_SUB(temp, temp2);

 		temp = MONT_MUL(BASE_FIELD_R8, S);
 		res.z = MONT_MUL(temp, S2);

		return res;
	}
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------

//test exponentiation: one operation per warp

//we do also implement Montgomery ladder algorithm

#define MONTGOMERY_LADDER(SUFFIX) \
DEVICE_FUNC ec_point ecc_mont_ladder_exp##SUFFIX(const ec_point& pt, const uint256_g& power)\
{\
    ec_point R0 = point_at_infty();\
    ec_point R1 = pt;\
    for (int i = N_BITLEN - 1; i >= 0; i--)\
    {\
        bool flag = get_bit(power, i);\
        ec_point& Q = (flag ? R0 : R1);\
        ec_point& T = (flag ? R1 : R0);\
\
        Q = ECC_ADD##SUFFIX(Q, R);\
        T = ECC_DOUBLE##SUFFIX(T);\
    }\
\
    return R0;\
}
    
MONTGOMERY_LADDER(_PROJ)
MONTGOMERY_LADDER(_JAC)

#define EXP_ONE_OP_PER_WRAP(func_name) \
__global__ void func_name##_kernel_per_warp(const ec_point* a_arr, const uint256_g* b_arr, ec_point* c_arr, size_t count)\
{\
    size_t tid = (threadIdx.x + blockIdx.x * blockDim.x;) / WARP_SIZE\
	while (tid < arr_len)\
	{\
		if (threadIdx.x % WARP_SIZE == 0)\
            c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += (blockDim.x * gridDim.x;) / WARP_SIZE\
	}\
}\
\
void func_name##_driver_per_warp(const ec_point* a_arr, const uint256_g* b_arr, ec_point* c_arr, size_t count)\
{\       
    cudaDeviceProp prop;\
    cudaGetDeviceProperties(&prop, 0);\
    uint32_t smCount = prop.multiProcessorCount;\   
    geometry2 geometry = find_geometry2(func_name##_kernel_per_warp, 0, uint32_t smCount);\
\
    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl;\
    func_name##_kernel_per_warp<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, count);\
}


