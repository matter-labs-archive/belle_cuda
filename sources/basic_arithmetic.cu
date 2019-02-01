#include "cuda_structs.h"

//128 bit addition & substraction:
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

//NB: https://devtalk.nvidia.com/default/topic/948014/forward-looking-gpu-integer-performance/?offset=14
//It seems CUDA has no natural support for 64-bit integer arithmetic (and 16 bit will be obviously toooooooo slow)

//in order to implement Karatsuba multiplication we need addition with carry!
//HOW TO GET VALUE OF CARRY FLAG!
//NO WAY! VERY DUMB STUPID NVIDIA PTX ASSEMBLY!

DEVICE_FUNC uint128_with_carry_g add_uint128_with_carry_asm(const uint128_g& lhs, const uint128_g& rhs)
{
    uint128_with_carry_g result;
		asm (	"add.cc.u32      %0, %5,  %9;\n\t"
         	 	"addc.cc.u32     %1, %6,  %10;\n\t"
         	 	"addc.cc.u32     %2, %7,  %11;\n\t"
         		"addc.cc.u32     %3, %8,  %12;\n\t"
                "addc.u32        %4, 0, 0;\n\t"
         		: "=r"(result.val.n[0]), "=r"(result.val.n[1]), "=r"(result.val.n[2]), "=r"(result.val.n[3]), "=r"(result.carry)
				: "r"(lhs.n[0]), "r"(lhs.n[1]), "r"(lhs.n[2]), "r"(lhs.n[3]),
				    "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]));

    return result;
}


DEVICE_FUNC uint128_g sub_uint128_asm(const uint128_g& lhs, const uint128_g& rhs)
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

//256 bit addition & substraction
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC uint256_g add_uint256_naive(const uint256_g& lhs, const uint256_g& rhs)
{
    uint32_t carry = 0;
    uint256_g result;
    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
    {
        result.n[i] = lhs.n[i] + rhs.n[i] + carry;
        carry = (result.n[i] < lhs.n[i]);
    }
    return result;
}

DEVICE_FUNC uint256_g add_uint256_asm(const uint256_g& lhs, const uint256_g& rhs)
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

DEVICE_FUNC uint256_g sub_uint256_naive(const uint256_g& lhs, const uint256_g& rhs)
{
    uint32_t borrow = 0;
    uint256_g result;
    
    #pragma unroll
	for (uint32_t i = 0; i < N; i++)
    
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

DEVICE_FUNC uint256_g sub_uint256_asm(const uint256_g& lhs, const uint256_g& rhs)
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

//NB: here addition and substraction is done in place!

DEVICE_FUNC void add_uint_uint256_asm(uint256_g& elem, uint32_t num)
{
    asm(    "add.cc.u32      %0, %0,  %8;\n\t"
            "addc.cc.u32     %1, %1,   0;\n\t"
            "addc.cc.u32     %2, %2,   0;\n\t"
            "addc.cc.u32     %3, %3,   0;\n\t"
            "addc.cc.u32     %4, %4,   0;\n\t"
            "addc.cc.u32     %5, %5,   0;\n\t"
            "addc.cc.u32     %6, %6,   0;\n\t"
            "addc.u32        %7, %7,   0;\n\t"
            : "+r"(elem.n[0]), "+r"(elem.n[1]), "+r"(elem.n[2]), "+r"(elem.n[3]),
                "+r"(elem.n[4]), "+r"(elem.n[5]), "+r"(elem.n[6]), "+r"(elem.n[7])
            : "r"(num));
}

DEVICE_FUNC void sub_uint_uint256_asm(uint256_g& elem, uint32_t num)
{
     asm(    "sub.cc.u32      %0, %0,  %8;\n\t"
            "subc.cc.u32     %1, %1,   0;\n\t"
            "subc.cc.u32     %2, %2,   0;\n\t"
            "subc.cc.u32     %3, %3,   0;\n\t"
            "subc.cc.u32     %4, %4,   0;\n\t"
            "subc.cc.u32     %5, %5,   0;\n\t"
            "subc.cc.u32     %6, %6,   0;\n\t"
            "subc.u32        %7, %7,   0;\n\t"
            : "+r"(elem.n[0]), "+r"(elem.n[1]), "+r"(elem.n[2]), "+r"(elem.n[3]),
                "+r"(elem.n[4]), "+r"(elem.n[5]), "+r"(elem.n[6]), "+r"(elem.n[7])
            : "r"(num));
}

//256 bit shifts
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC uint256_g shift_right_asm(const uint256_g& elem, uint32_t shift)
{
    uint256_g result;

    asm (   ".reg .u32 temp;\n\t"
#if (__CUDA_ARCH__ < 500)
            ".reg .u32 x, y;\n\t"
#endif
            "mov.u32 x, %16;\n\t"
            
#if (__CUDA_ARCH__ >= 500)
            "shf.r.clamp.b32 %8, %0, %1, x;\n\t"
            "shf.r.clamp.b32 %9, %1, %2, x;\n\t"
            "shf.r.clamp.b32 %10, %2, %3, x;\n\t"
            "shf.r.clamp.b32 %11, %3, %4, x;\n\t"
            "shf.r.clamp.b32 %12, %4, %5, x;\n\t"
            "shf.r.clamp.b32 %13, %5, %6, x;\n\t"
            "shf.r.clamp.b32 %14, %6, %7, x;\n\t"
            "shr.b32 %15, %7, x;\n\t"
#else
            "sub.u32 y, 32, x;\n\t"

            "shr.b32 %8, %0, x;\n\t"
            "shl.b32 temp, %1, y;\n\t"
            "or.b32 %8, %8, temp;\n\t"

            "shr.b32 %9, %1, x;\n\t"
            "shl.b32 temp, %2, y;\n\t"
            "or.b32 %9, %9, temp;\n\t"

            "shr.b32 %10, %2, x;\n\t"
            "shl.b32 temp, %3, y;\n\t"
            "or.b32 %10, %10, temp;\n\t"

            "shr.b32 %11, %3, x;\n\t"
            "shl.b32 temp, %4, y;\n\t"
            "or.b32 %11, %11, temp;\n\t"

            "shr.b32 %12, %4, x;\n\t"
            "shl.b32 temp, %5, y;\n\t"
            "or.b32 %12, %12, temp;\n\t"

            "shr.b32 %13, %5, x;\n\t"
            "shl.b32 temp, %6, y;\n\t"
            "or.b32 %13, %13, temp;\n\t"

            "shr.b32 %14, %6, x;\n\t"
            "shl.b32 temp, %7, y;\n\t"
            "or.b32 %14, %14, temp;\n\t"

            "shr.b32 %15, %7, x;\n\t"
#endif
           : "=r"(result.n[0]), "=r"(result.n[1]), "=r"(result.n[2]), "=r"(result.n[3]),
				  "=r"(result.n[4]), "=r"(result.n[5]), "=r"(result.n[6]), "=r"(result.n[7])
				 : "r"(elem.n[0]), "r"(elem.n[1]), "r"(elem.n[2]), "r"(elem.n[3]),
				  "r"(elem.n[4]), "r"(elem.n[5]), "r"(elem.n[6]), "r"(elem.n[7]), "r"(shift));
    
    return result;
}


DEVICE_FUNC uint256_g shift_left_asm(const uint256_g& elem, uint32_t shift)
{
    uint256_g result;

    asm (   ".reg .u32 temp;\n\t"
#if (__CUDA_ARCH__ < 500)
            ".reg .u32 x, y;\n\t"
#endif
            "mov.u32 x, %16;\n\t"
            
#if (__CUDA_ARCH__ >= 500)
            "shf.l.clamp.b32 %15, %6, %7, x;\n\t"
            "shf.l.clamp.b32 %14, %5, %6, x;\n\t"
            "shf.l.clamp.b32 %13, %4, %5, x;\n\t"
            "shf.l.clamp.b32 %12, %3, %4, x;\n\t"
            "shf.l.clamp.b32 %11, %2, %3, x;\n\t"
            "shf.l.clamp.b32 %10, %1, %2, x;\n\t"
            "shf.l.clamp.b32 %9, %0, %1, x;\n\t"
            "shl.b32 %8, %0, x;\n\t"
#else
            "sub.u32 y, 32, x;\n\t"
            "shl.b32 %8, %0, x;\n\t"

            "shl.b32 %9, %1, x;\n\t"
            "shr.b32 temp, %0, y;\n\t"
            "or.b32 %9, %9, temp;\n\t"

            "shl.b32 %10, %2, x;\n\t"
            "shr.b32 temp, %1, y;\n\t"
            "or.b32 %10, %10, temp;\n\t"

            "shl.b32 %11, %3, x;\n\t"
            "shr.b32 temp, %2, y;\n\t"
            "or.b32 %9, %9, temp;\n\t"

            "shl.b32 %12, %4, x;\n\t"
            "shr.b32 temp, %3, y;\n\t"
            "or.b32 %12, %12, temp;\n\t"

            "shl.b32 %13, %5, x;\n\t"
            "shr.b32 temp, %4, y;\n\t"
            "or.b32 %13, %13, temp;\n\t"

            "shl.b32 %14, %6, x;\n\t"
            "shr.b32 temp, %5, y;\n\t"
            "or.b32 %14, %14, temp;\n\t"

            "shl.b32 %15, %7, x;\n\t"
            "shr.b32 temp, %6, y;\n\t"
            "or.b32 %15, %15, temp;\n\t"           
#endif
           : "=r"(result.n[0]), "=r"(result.n[1]), "=r"(result.n[2]), "=r"(result.n[3]),
				  "=r"(result.n[4]), "=r"(result.n[5]), "=r"(result.n[6]), "=r"(result.n[7])
				 : "r"(elem.n[0]), "r"(elem.n[1]), "r"(elem.n[2]), "r"(elem.n[3]),
				  "r"(elem.n[4]), "r"(elem.n[5]), "r"(elem.n[6]), "r"(elem.n[7]), "r"(shift));
    
    return result;
}


//256 bit  comparison and zero equality
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC int cmp_uint256_naive(const uint256_g& lhs, const uint256_g& rhs)
{
    #pragma unroll
    for (int32_t i = N -1 ; i >= 0; i--)
    {
        if (lhs.n[i] > rhs.n[i])
            return 1;
        else if (lhs.n[i] < rhs.n[i])
            return -1;
    }
    return 0;
}

DEVICE_FUNC bool is_zero(const uint256_g& x)
{
    #pragma unroll
    for (int32_t i = N -1 ; i >= 0; i--)
    {
        if (x.n[i] != 0)
            return false;
    }
    return true;
}

DEVICE_FUNC bool is_even(const uint256_g& x)
{
    return  !CHECK_BIT(x.n[0], 0);
}


