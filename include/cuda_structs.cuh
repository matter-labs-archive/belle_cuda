#ifndef CUDA_STRUCTS_CUH
#define CUDA_STRUCTS_CUH

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define DEVICE_FUNC __device__
#else
#define DEVICE_FUNC
#endif

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