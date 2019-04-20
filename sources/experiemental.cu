#include "cuda_structs.h"

#include <iostream>

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
#define ROUND_UP(n,d) (( (n) + (d) - 1 ) / (d) * (d))
#define DIV_ROUND_UP(n,d) (( (n) + (d) - 1)/ (d))
#define MIN(x, y) (( (x) < (y) ? (x) : (y) ))

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
        for(uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < count; j += blockDim.y * gridDim.y)
        {
            out[j * limbs + i] = in[ i * stride + j];
        }
    }
}

inline void xmpS2C(uint32_t count, const uint32_t * in, uint32_t * out)
{
    dim3 threads, blocks;
    uint32_t limbs = N;
    uint32_t stride = ROUND_UP(count, 32);
    //round up to 128 bше boundarн

    //target 128 threads
    threads.x = MIN(32, limbs);
    threads.y = MIN(DIV_ROUND_UP(128, threads.x), count);

    blocks.x=DIV_ROUND_UP(count, threads.x);
    blocks.y=DIV_ROUND_UP(limbs, threads.y);

    //convert from climbs to slimbs
    xmpS2C_kernel<<<blocks, threads>>>(count, limbs, stride, in, out);
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
        for(uint32_t index = 0; index < N; index++)\
        {\
            A.n[index] = a_data[index * stride];\
            B.n[index] = b_data[index * stride];\
        }\
\
        func_name(A, B, C);\
\
        for(uint32_t index = 0; index < N; index++)\
        {\
            c_data[index * stride] = C.n[index];\
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

DEVICE_FUNC __inline__ void long_in_place_add(uint32_t& A, uint32_t B, uint32_t& carry)
{
    asm(    "{\n\t"  
            "add.cc.u32 %0, %0, %2;\n\t"
            "addc.u32 %1, 0, 0;}\n\t" 
            : "+r"(A), "=r"(carry) : "r"(B));
}

DEVICE_FUNC __inline__ void long_in_place_sub(uint32_t& A, uint32_t B, uint32_t& carry)
{
    asm(    "{\n\t"  
            "sub.cc.u32 %0, %0, %2;\n\t"
            "addc.u32 %1, 0, 0;\n\t"
            "xor.b32 %1, %1, 0x1;}\n\t" 
            : "+r"(A), "=r"(carry) : "r"(B));
}

#define THREADS_PER_MUL 8

DEVICE_FUNC uint64_g warp_based_mul_naive(uint32_t A, const uint32_t B)
{
    uint32_t lane = threadIdx.x % THREADS_PER_MUL;
    uint32_t u = 0, v = 0, c = 0;

    for (uint32_t j = 0; j < THREADS_PER_MUL; j++)
    {
        uint32_t b = __shfl_sync(0xffffffff, B, j, THREADS_PER_MUL);
        uint32_t t;

        asm(    "mad.lo.cc.u32 %0, %3, %4, %0;\n\t" 
                "madc.hi.cc.u32 %1, %3, %4, %1;\n\t"
                "addc.u32 %2, 0, 0;\n\t"
                : "+r"(v), "+r"(u), "=r"(t)
                : "r"(A), "r"(b));

        v = __shfl_sync(0xffffffff, v, (lane + 1) % THREADS_PER_MUL, THREADS_PER_MUL);
        c = __shfl_sync(0xffffffff, c, (lane + 1) % THREADS_PER_MUL, THREADS_PER_MUL);

        if (lane == THREADS_PER_MUL - 1)
        {
            c = v;
            v = 0;
        }

        asm(   "{\n\t"  
                "add.cc.u32 %1, %0, %1;\n\t"
                "addc.u32 %0, %2, 0;}\n\t" 
                : "+r"(u), "+r"(v) : "r"(t));
    }

    while (__any_sync(0xffffffff, u != 0))
    {
        u = __shfl_sync(0xffffffff, u, (lane + THREADS_PER_MUL - 1) % THREADS_PER_MUL, THREADS_PER_MUL);
        long_in_place_add(v, u, u);
    }

    uint64_g res;
    res.low = c;
    res.high = v;
    return res;
}


//I bet there must be a faster solution!
//TODO: have a look at https://ieeexplore.ieee.org/document/5669278

__global__ void warp_based_mul_naive_kernel(const uint256_g* a_arr, const uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t idx = tid / THREADS_PER_MUL;
    size_t lane = tid % THREADS_PER_MUL;
	while (idx < arr_len)
	{
		uint32_t A = a_arr[idx].n[lane];
        uint32_t B = b_arr[idx].n[lane];
        uint64_g C = warp_based_mul_naive(A, B);

        c_arr[idx].n[lane] = C.low;
        c_arr[idx].n[lane + N] = C.high;

		idx += (blockDim.x * gridDim.x) / THREADS_PER_MUL;
	}
}

void warp_based_mul_naive_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t smCount = prop.multiProcessorCount;   
    geometry2 geometry = find_geometry2(warp_based_mul_naive_kernel, 0, smCount);

    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl;
    warp_based_mul_naive_kernel<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

#define WARP_MUL(a, b) warp_based_mul_naive(a, b)


DEVICE_FUNC __inline__ bool warp_based_geq(uint32_t A, uint32_t B, uint32_t mask, uint32_t warp_mask)
{
    uint32_t cond1 = __ballot_sync(mask, A > B);
    uint32_t cond2 = __ballot_sync(mask, A < B);
    return (__clz(cond1 & warp_mask) <= __clz(cond2 & warp_mask));
}

//we assume that addition is without overflow
DEVICE_FUNC __inline__ uint32_t warp_based_add(uint32_t A, uint32_t B, uint32_t mask, uint32_t lane)
{
    uint32_t carry;
    long_in_place_add(A, B, carry);

    //propagate carry
    while (__any_sync(mask, carry != 0))
    {
        uint32_t x = __shfl_sync(mask, carry, (lane + THREADS_PER_MUL - 1) % THREADS_PER_MUL, THREADS_PER_MUL);
        long_in_place_add(A, x, carry);
    }
    return A;
}

//we assume that A >= B
DEVICE_FUNC __inline__ uint32_t warp_based_sub(uint32_t A, uint32_t B, uint32_t mask, uint32_t lane)
{
    uint32_t carry;
    long_in_place_sub(A, B, carry);
   
    //propagate borrow
    while (__any_sync(mask, carry != 0))
    {
        uint32_t x = __shfl_sync(mask, carry, (lane + THREADS_PER_MUL - 1) % THREADS_PER_MUL, THREADS_PER_MUL);
       
        long_in_place_sub(A, x, carry);
    }
    return A;
}

DEVICE_FUNC uint32_t mont_mul_warp_based(uint32_t A, uint32_t B, uint32_t warp_mask, uint32_t lane)
{
    //LARGE REDC Montgomety mul
    // T = A * B
    //m = ((T mod R) * N) mod R
    //t = (T + m * p) / R
    //if t >= N then t = t - N
   
    uint64_g temp1 = WARP_MUL(A, B);
    uint64_g temp2 = WARP_MUL(temp1.low, BASE_FIELD_N_LARGE.n[lane]);
    temp2 = WARP_MUL(temp2.low, BASE_FIELD_P.n[lane]);

    //adding higher 8-words part
    uint32_t carry = 0;
    long_in_place_add(temp1.high, temp2.high, carry);

    //we are going to check if there is overflow from lower 8-words part
    uint32_t sum = temp1.low + temp2.low;
    uint32_t cond1_mask = __ballot_sync(0xffffffff, sum < temp1.low) & warp_mask;
    uint32_t cond2_mask = __ballot_sync(0xffffffff, sum == 0xffffffff) & warp_mask;

    if (lane == THREADS_PER_MUL - 1)
    {
        //TODO: here should be a vey intimate check that I'm unable to perform
        carry = 0;
    }

    //propagate carry
    while (__any_sync(0xffffffff, carry != 0))
    {
        uint32_t x = __shfl_sync(0xffffffff, carry, (lane + THREADS_PER_MUL - 1) % THREADS_PER_MUL, THREADS_PER_MUL);
        long_in_place_add(temp1.high, x, carry);
    }

    //now temp1.high holds t, compare t with N:
    if (warp_based_geq(temp1.high, BASE_FIELD_P.n[lane], 0xffffffff, warp_mask))
        warp_based_sub(temp1.high, BASE_FIELD_P.n[lane], 0xffffffff, lane);

    return temp1.high;
}

DEVICE_FUNC uint32_t mont_mul_warp_based_ver2(uint32_t A, uint32_t B, uint32_t warp_mask, uint32_t lane)
{
    uint32_t S[3] = {0, 0, 0};
    uint32_t P = BASE_FIELD_P.n[lane];

    for (uint32_t j = 0; j < 8; j++)
    {
        uint32_t b = __shfl_sync(0xffffffff, B, j, THREADS_PER_MUL);

        asm(    "mad.lo.cc.u32 %0, %3, %4, %0;\n\t" 
                "madc.hi.cc.u32 %1, %3, %4, %1;\n\t"
                "addc.u32 %2, %2, 0;\n\t"
                : "+r"(S[0]), "+r"(S[1]), "+r"(S[2])
                : "r"(A), "r"(b));

        uint32_t q;
        if (lane == 0)
            q = S[0] * BASE_FIELD_N;
        q = __shfl_sync(0xffffffff, q, 0, THREADS_PER_MUL);
        
        asm(    "mad.lo.cc.u32 %0, %3, %4, %0;\n\t" 
                "madc.hi.cc.u32 %1, %3, %4, %1;\n\t"
                "addc.u32 %2, %2, 0;\n\t"
                : "+r"(S[0]), "+r"(S[1]), "+r"(S[2])
                : "r"(q), "r"(P));

        uint32_t temp = __shfl_down_sync(0xffffffff, S[0], 1, THREADS_PER_MUL);
        if (lane == THREADS_PER_MUL - 1)
            temp = 0;

        asm(   "{\n\t"  
                "add.cc.u32 %0, %1, %3;\n\t"
                "addc.u32   %1, %2, 0;}\n\t" 
                : "=r"(S[0]), "+r"(S[1]) : "r"(S[2]), "r"(temp));

        S[2] = 0;
    }

    //propagate carry (carry is located in S1)
    while (__any_sync(0xffffffff, S[1] != 0))
    {
        uint32_t x = __shfl_sync(0xffffffff, S[1], (lane + THREADS_PER_MUL - 1) % THREADS_PER_MUL, THREADS_PER_MUL);
        long_in_place_add(S[0], x, S[1]);
    }

    //NB: we can chain several montgomety muls without the below reduction
    //It also results in no warp divergence!
   
    if (warp_based_geq(S[0], P, 0xffffffff, warp_mask))
    {
        warp_based_sub(S[0], P, 0xffffffff, lane);
    }

    return S[0];
}

#define LOOP_UNROLLER(idx) \
"shfl.sync.idx.b32  b, B, "#idx", 0x181f, 0xffffffff;\n\t" \
"mad.lo.cc.u32 S0, A, b, S0;\n\t" \
"madc.hi.cc.u32 S1, A, b, S1;\n\t" \
"addc.u32 S2, S2, 0;\n\t" \
\
"mul.lo.u32 q, S0, N;\n\t" \
"shfl.sync.idx.b32  q, q, 0, 0x181f, 0xffffffff;\n\t" \
\
"mad.lo.cc.u32 S0, P, q, S0;\n\t" \
"madc.hi.cc.u32 S1, P, q, S1;\n\t" \
"addc.u32 S2, S2, 0;\n\t" \
\
"shfl.sync.down.b32  temp|cond, S0, 1, 0x181f, 0xffffffff;\n\t" \
"@!cond mov.u32 temp, 0;\n\t" \
\
"add.cc.u32 S0, S1, temp;\n\t" \
"addc.u32 S1, S2, 0;\n\t" \
"mov.u32 S2, 0;\n\t"


DEVICE_FUNC uint32_t mont_mul_warp_based_asm(uint32_t A, uint32_t B, uint32_t warp_mask, uint32_t lane)
{
    uint32_t result;
    
    asm(    "{\n\t"
            ".reg .u32 A, B, b, S<3>;\n\t"
            ".reg .u32 q, P, N, temp;\n\t"
            ".reg .pred cond;\n\t"
            ".reg .b64 base_addr, addr;\n\t"

            "mov.u32 A, %1;\n\t"
            "mov.u32 B, %2;\n\t"

            "ld.const.u32 N, [BASE_FIELD_N];\n\t"
            "mov.u64 base_addr, BASE_FIELD_P;\n\t"
            "mul.wide.u32 addr, %3, 4;\n\t"            	
	        "add.s64 addr, base_addr, addr;\n\t"
	        "ld.const.u32 P, [addr];\n\t"

            "mov.u32 S0, 0;\n\t"
            "mov.u32 S1, 0;\n\t"
            "mov.u32 S2, 0;\n\t"
            
            LOOP_UNROLLER(0)
            LOOP_UNROLLER(1)
            LOOP_UNROLLER(2)
            LOOP_UNROLLER(3)
            LOOP_UNROLLER(4)
            LOOP_UNROLLER(5)
            LOOP_UNROLLER(6)
            LOOP_UNROLLER(7)

            "L1:\n\t"
            "setp.ne.u32 cond, S1, 0;\n\t"
            "vote.sync.any.pred cond, cond, 0xffffffff;\n\t"
            "@!cond bra L2;\n\t"
            "shfl.sync.up.b32  temp|cond, S1, 0x1, 0x1800, 0xffffffff;\n\t"
            "@!cond mov.u32 temp, 0;\n\t" 
            "add.cc.u32 S0, S0, temp;\n\t"
            "addc.u32 S1, 0, 0;\n\t" 
            "bra L1;\n\t"

            "L2:\n\t"
            "mov.u32 %0, S0;}\n\t"
            : "=r"(result) : "r"(A), "r"(B), "r"(lane));

    //final comparison
    //todo: rewrite it in ASM PTX!
    if (warp_based_geq(result, BASE_FIELD_P.n[lane], 0xffffffff, warp_mask))
    {
        warp_based_sub(result, BASE_FIELD_P.n[lane], 0xffffffff, lane);
    }

    return result;
}

#define WRAP_BASED_MONT_MUL(func_name) \
__global__ void func_name##_kernel(const uint256_g* a_arr, const uint256_g* b_arr, uint256_g* c_arr, size_t arr_len) \
{\
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x; \
    size_t idx = tid / THREADS_PER_MUL; \
    uint32_t lane = tid % THREADS_PER_MUL; \
    uint32_t shift = (tid % WARP_SIZE ) / THREADS_PER_MUL; \
    uint32_t warp_mask = 0xff << (shift * THREADS_PER_MUL); \
\
	while (idx < arr_len) \
	{ \
		uint32_t A = a_arr[idx].n[lane]; \
        uint32_t B = b_arr[idx].n[lane]; \
        uint32_t C = func_name(A, B, warp_mask, lane); \
\
        c_arr[idx].n[lane] = C; \
		idx += (blockDim.x * gridDim.x) / THREADS_PER_MUL; \
	} \
} \
\
void func_name##_driver(uint256_g* a_arr, uint256_g* b_arr, uint256_g* c_arr, size_t arr_len) \
{\
    cudaDeviceProp prop; \
    cudaGetDeviceProperties(&prop, 0); \
    uint32_t smCount = prop.multiProcessorCount; \
    geometry2 geometry = find_geometry2(func_name##_kernel, 0, smCount); \
\
    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl; \
    func_name##_kernel<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, arr_len); \
}

WRAP_BASED_MONT_MUL(mont_mul_warp_based)
WRAP_BASED_MONT_MUL(mont_mul_warp_based_ver2)
WRAP_BASED_MONT_MUL(mont_mul_warp_based_asm)

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
        Q = ECC_ADD##SUFFIX(Q, T);\
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


//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------

//warp-based elliptic curve point addition

//16 threads are used to calculate one operation
#define THREADS_PER_ECC_ADD 16

DEVICE_FUNC __inline__ bool is_leader_lane()
{
    return (threadIdx.x % THREADS_PER_ECC_ADD == 0);
}

//chose an element based on subwarp
DEVICE_FUNC __inline__ uint32_t subwarp_choose_elem(const uint256_g& X, const uint256_g& Y, uint32_t lane, bool choice)
{
    uint256& temp = (choice ? X : Y);
    return temp.n[lane];
}

DEVICE_FUNC __inline__ uint32_t subwarp_choose_elem(uint32_t X, uint32_t Y, bool choice)
{
    return (choice ? X : Y);
}

DEVICE_FUNC __inline__ uint32_t warp_based_field_add(uint32_t A, uint32_t B, uint32_t mask, uint32_t lane)
{
    uint32_t temp = warp_based_add(A, B, mask, lane);
    if (warp_based_geq(temp, BASE_FIELD_P.n[lane], mask))
    {
        return warp_based_sub(temp, BASE_FIELD_P.n[lane], mask);
    }
    return temp;
}

DEVICE_FUNC __inline__ uint32_t warp_based_field_sub(uint32_t A, uint32_t B, uint32_t mask, uint32_t lane)
{
    if (warp_based_geq(A, B, mask, warp_idx))
    {
        return warp_based_sub(A, B, mask);
    }
    else
    {
        uint32_t temp = warp_based_add(A, BASE_FIELD_P.n[lane]);
        return warp_based_sub(temp, B, mask);
    }
}
else
DEVICE_FUNC __inline__ uint32_t subwarps_exchange_elems(uint32_t A, uint32_t B, bool rrs)
{
    uint32_t elem = (rrs ? A : B);
    return __shfl_down_sync(0xFFFFFFFF, elem, THREADS_PER_MUL, THREADS_PER_ECC_ADD);
}

DEVICE_FUNC void ECC_add_proj_warp_based(const ec_point& left, const ec_point& right, ec_point& OUT, uint32_t lane, uint32_t subwarp_mask)
{
    uint32_t exit_flag = 0;
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t warp_idx = tid / THREADS_PER_MUL;
    uint32_t mask = (THREADS_PER_MUL - 1) << (warp_idx * THREADS_PER_MUL);
    uint32_t lane = tid % THREADS_PER_MUL;
    bool rrs = (warp_idx % THREADS_PER_ECC_ADD) < (THREADS_PER_ECC_ADD / 2 );

    if (is_leader_lane())
    {
        if (is_infinity(A))
        {
            C = B;
            exit_flag = 1;
        }
	    else if (is_infinity(B))
        {
		    C = A;
            exit_flag = 1;
        }
    }

    exit_flag = __shfl_sync(0xFFFFFFFF, exit_flag, 0, THREADS_PER_ECC_ADD);
    if (exit_flag)
        return;

    uint32_t Z = subwarp_chooser(left.z, right.z, lane, rrs);
    uint32_t Y = subwarp_chooser(right.y, left.y);
    uint32_t U12 = mont_mul_warp_based(Z, Y);

    uint32_t X = subwarp_chooser(right.x, left.x);
    uint32_t V12 = mont_mul_warp_based(Z, X);

    uint32_t U_V = subwarp_chooser(U12, V12, rrs);
    uint32_t temp = warp_based_exchanger(V12, U12, rrs);

    U_V = warp_based_field_sub(U_V, temp, mask, warp_idx, lane);
    

    //check for equality

    //squaring of U and V:
    uint32_t U_V_sq = mont_mul_warp_based(U_V, U_V);

    temp = warp_based_exchanger(V12, U12, rrs);
   
	uint256_g Vcube = MONT_MUL(Vsq, V);
	uint256_g W = MONT_MUL(left.z, right.z);

	temp1 = MONT_MUL(temp1, W);
    temp2 = MONT_MUL(BASE_FIELD_R2, Vsq);

    temp2 = MONT_MUL(temp2, V2);
    res.z = MONT_MUL(Vcube, W);

    tempx = MONT_MUL(Vsq, V2);
    temp3 = MONT_MUL(Vcube, U2);

    //without pair
    temp1 = FIELD_SUB(temp1, Vcube);
    uint256_g A = FIELD_SUB(temp1, temp2);
    tempx = FIELD_SUB(tempx, A);

	res.x = MONT_MUL(V, A);	
	tempx = MONT_MUL(U, tempx);
	
	res.y = FIELD_SUB(tempx, temp3);

	
	return res;

}

__global__ void ECC_add_proj_warp_based_kernel(const ec_point* a_arr, const ec_point* b_arr, ec_point *c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x);
    size_t idx = tid / THREADS_PER_ECC_ADD;
    uint32_t shift = (tid % WARP_SIZE ) / THREADS_PER_ECC_ADD; 
    uint32_t subwarp_mask = 0xffff << (shift * THREADS_PER_ECC_ADD);
    uint32_t lane = tid % THREADS_PER_ECC_ADD;

	while (idx < arr_len)
	{
		ECC_add_proj_warp_based(a_arr[idx], b_arr[idx], c_arr[idx], lane, 0xffff);
		idx += (blockDim.x * gridDim.x) / THREADS_PER_ECC_ADD;
	}
}

void ECC_add_proj_warp_based_driver(const ec_point* a_arr, const ec_point* b_arr, ec_point* c_arr, size_t arr_len)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t smCount = prop.multiProcessorCount;   
    geometry2 geometry = find_geometry2(ECC_add_proj_warp_based_kernel, 0, uint32_t smCount);

    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl;
    ECC_add_proj_warp_based_kernel<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

