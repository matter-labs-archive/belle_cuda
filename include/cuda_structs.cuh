#ifndef CUDA_STRUCTS_CUH
#define CUDA_STRUCTS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define DEVICE_FUNC __device__
#define DEVICE_VAR __device__
#define CONST_MEMORY __constant__
#else
#define DEVICE_FUNC
#define DEVICE_VAR
#define CONST_MEMORY
#endif

#define HALF_N 4
#define N 8
#define N_DOUBLED 16

#include <stdint.h>

struct uint64_g
{
    uint32_t n[2];
};

struct uint128_g
{
    union
    {
        uint32_t n[4];
        struct
        {
            uint64_t low;
            uint64_t high;
        };
    };
};

struct uint128_with_carry_g
{
    uint128_g val;
    uint32_t carry;
};

//NB: may be this should somehow help?
//https://stackoverflow.com/questions/10297067/in-a-cuda-kernel-how-do-i-store-an-array-in-local-thread-memory

struct uint256_g
{
    union
    {
        uint32_t n[8];
        uint64_t nn[4];
        struct
        {
            uint128_g low;
            uint128_g high;
        };
    };
};

struct uint512_g
{
    union
    {
        uint32_t n[16];
        uint64_t nn[8];
    };
};

#endif