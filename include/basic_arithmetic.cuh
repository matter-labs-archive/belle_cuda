#ifndef BASIC_ARITHM_CUH
#define BASIC_ARITHM_CUH

#include "cuda_structs.cuh"

//128 bit addition & substraction:
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

//NB: https://devtalk.nvidia.com/default/topic/948014/forward-looking-gpu-integer-performance/?offset=14
//It seems CUDA has no natural support for 64-bit integer arithmetic

DEVICE_FUNC inline uint128_g add_uint128_asm(const uint128_g& lhs, const uint128_g& rhs)
{
    uint128_g result;
		asm (	"add.cc.u32      %0, %4,  %8;\n\t"
         	 	"addc.cc.u32     %1, %5,  %9;\n\t"
         	 	"addc.cc.u32     %2, %6,  %10;\n\t"
         		"addc.u32        %3, %7,  %11;\n\t"
         		: "=r"(result.n[0]), "=r"(result.n[1]), "=r"(result.n[2]), "=r"(result.n[3])
				: "r"(lhs.n[0]), "r"(lhs.n[1]), "r"(lhs.n[2]), "r"(lhs.n[3]),
				    "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]));

    return result;
}


DEVICE_FUNC inline uint128_g sub_uint128_asm(const uint128_g& lhs, const uint128_g& rhs)
{
    uint128_g result;
		asm (	"sub.cc.u32      %0, %4,  %8;\n\t"
         	 	"subc.cc.u32     %1, %5,  %9;\n\t"
         	 	"subc.cc.u32     %2, %6,  %10;\n\t"
         		"subc.u32        %3, %7,  %11;\n\t"
         		: "=r"(result.n[0]), "=r"(result.n[1]), "=r"(result.n[2]), "=r"(result.n[3])
				: "r"(lhs.n[0]), "r"(lhs.n[1]), "r"(lhs.n[2]), "r"(lhs.n[3]),
				    "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]));

    return result;
}

//256 bit addition & substraction & comparison
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC inline uint256_g add_uint256_naive(const uint256_g& lhs, const uint256_g& rhs)
{
    uint32_t carry = 0;
    uint256_g result;
    #pragma unroll
    for (uint32_t i = 0; i < 8; i++)
    {
        result.n[i] = lhs.n[i] + rhs.n[i] + carry;
        carry = (result.n[i] < lhs.n[i]);
    }
    return result;
}

DEVICE_FUNC inline uint256_g add_uint256_asm(const uint256_g& lhs, const uint256_g& rhs)
{
    uint256_g result;
		asm (	"add.cc.u32      %0, %8,  %16;\n\t"
         	 	"addc.cc.u32     %1, %9,  %17;\n\t"
         	 	"addc.cc.u32     %2, %10, %18;\n\t"
         		"addc.cc.u32     %3, %11, %19;\n\t"
				"addc.cc.u32     %4, %12, %20;\n\t"
         		"addc.cc.u32     %5, %13, %21;\n\t"
         		"addc.cc.u32     %6, %14, %22;\n\t"
         		"addc.u32        %7, %15, %23;\n\t"
         		: "=r"(result.n[0]), "=r"(result.n[1]), "=r"(result.n[2]), "=r"(result.n[3]),
				    "=r"(result.n[4]), "=r"(result.n[5]), "=r"(result.n[6]), "=r"(result.n[7])
				: "r"(lhs.n[0]), "r"(lhs.n[1]), "r"(lhs.n[2]), "r"(lhs.n[3]),
				    "r"(lhs.n[4]), "r"(lhs.n[5]), "r"(lhs.n[6]), "r"(lhs.n[7]),
				    "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]),
				    "r"(rhs.n[4]), "r"(rhs.n[5]), "r"(rhs.n[6]), "r"(rhs.n[7]));

    return result;
}

DEVICE_FUNC inline uint256_g sub_uint256_naive(const uint256_g& lhs, const uint256_g& rhs)
{
    uint32_t borrow = 0;
    uint256_g result;
    
    #pragma unroll
	for (uint32_t i = 0; i < 8; i++)
    
    {
        uint32_t a = lhs.n[i], b = rhs.n[i];
        result.n[i] = a - borrow;
        if (b == 0)
        {				
            borrow = ( result.n[i] > a ? 1 : 0);
        }
        else
        {
            result.n[i] -= b;
            borrow = ( result.n[i] >= a ? 1 : 0);
        }
    }
  
    return result;	
}

DEVICE_FUNC inline uint256_g sub_uint256_asm(const uint256_g& lhs, const uint256_g& rhs)
{
    uint256_g result;

    asm (	    "sub.cc.u32      %0, %8,  %16;\n\t"
         	 	"subc.cc.u32     %1, %9,  %17;\n\t"
         	 	"subc.cc.u32     %2, %10, %18;\n\t"
         		"subc.cc.u32     %3, %11, %19;\n\t"
				"subc.cc.u32     %4, %12, %20;\n\t"
         		"subc.cc.u32     %5, %13, %21;\n\t"
         		"subc.cc.u32     %6, %14, %22;\n\t"
         		"subc.u32        %7, %15, %23;\n\t"
         		 : "=r"(result.n[0]), "=r"(result.n[1]), "=r"(result.n[2]), "=r"(result.n[3]),
				  "=r"(result.n[4]), "=r"(result.n[5]), "=r"(result.n[6]), "=r"(result.n[7])
				 : "r"(lhs.n[0]), "r"(lhs.n[1]), "r"(lhs.n[2]), "r"(lhs.n[3]),
				  "r"(lhs.n[4]), "r"(lhs.n[5]), "r"(lhs.n[6]), "r"(lhs.n[7]),
				  "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]),
				  "r"(rhs.n[4]), "r"(rhs.n[5]), "r"(rhs.n[6]), "r"(rhs.n[7]));
		
    return result;
}

DEVICE_FUNC inline int cmp_uint256_naive(const uint256_g& lhs, const uint256_g& rhs)
{
    #pragma unroll
    for (int32_t i = 7; i >= 0; i--)
    {
        if (lhs.n[i] > rhs.n[i])
            return 1;
        else if (lhs.n[i] < rhs.n[i])
            return -1;
    }
    return 0;
}

#endif