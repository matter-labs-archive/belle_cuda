#ifndef MUL_256_to_512_CUH
#define MUL_256_to_512_CUH

#include "mul_128_to_256.cuh"

DEVICE_FUNC inline uint512_g mul_uint256_to_512_naive(const uint256_g& u, const uint256_g& v)
{
    uint512_g w;
    #pragma unroll
	for (uint32_t j = 0; j < 16; j++)
        w.n[j] = 0;
		
    #pragma unroll
	for (uint32_t j = 0; j < 8; j++)
	{
        uint32_t k = 0;

        #pragma unroll
        for (uint32_t i = 0; i < 8; i++)
        {
            uint32_t high_word = 0;
            uint32_t low_word = 0;
            low_word = device_long_mul(u.n[i], v.n[j], &high_word);
            low_word = device_fused_add(low_word, w.n[i + j], &high_word);
            low_word = device_fused_add(low_word, k, &high_word);
            k = high_word;
            w.n[i + j] = low_word;
        }

        w.n[8 + j] = k;
    }

    return w;	
}

DEVICE_FUNC inline uint512_g mul_uint256_to_512_asm(const uint256_g& lhs, const uint256_g& rhs)
{
    uint512_g w;
		
    asm (   "mul.lo.u32   %0, %16, %24;\n\t"
            "mul.hi.u32   %1, %16, %24;\n\t"
            "mad.lo.cc.u32   %1, %17, %24, %1;\n\t"
            "madc.hi.u32   %2, %17, %24, 0;\n\t"
            "mad.lo.cc.u32   %1, %16, %25, %1;\n\t"
            "madc.hi.cc.u32   %2, %16, %25, %2;\n\t"
            "madc.hi.u32   %3, %18, %24, 0;\n\t"
            "mad.lo.cc.u32   %2, %18, %24, %2;\n\t"
            "madc.hi.cc.u32   %3, %17, %25, %3;\n\t"
            "madc.hi.u32   %4, %19, %24, 0;\n\t"
            "mad.lo.cc.u32   %2, %17, %25, %2;\n\t"
            "madc.hi.cc.u32   %3, %16, %26, %3;\n\t"
            "madc.hi.cc.u32   %4, %18, %25, %4;\n\t"
            "madc.hi.u32   %5, %20, %24, 0;\n\t"
            "mad.lo.cc.u32   %2, %16, %26, %2;\n\t"
            "madc.lo.cc.u32   %3, %19, %24, %3;\n\t"
            "madc.hi.cc.u32   %4, %17, %26, %4;\n\t"
            "madc.hi.cc.u32   %5, %19, %25, %5;\n\t"
            "madc.hi.u32   %6, %21, %24, 0;\n\t"
            "mad.lo.cc.u32   %3, %18, %25, %3;\n\t"
            "madc.hi.cc.u32   %4, %16, %27, %4;\n\t"
            "madc.hi.cc.u32   %5, %18, %26, %5;\n\t"
            "madc.hi.cc.u32   %6, %20, %25, %6;\n\t"
            "madc.hi.u32   %7, %22, %24, 0;\n\t"
            "mad.lo.cc.u32   %3, %17, %26, %3;\n\t"
            "madc.lo.cc.u32   %4, %20, %24, %4;\n\t"
            "madc.hi.cc.u32   %5, %17, %27, %5;\n\t"
            "madc.hi.cc.u32   %6, %19, %26, %6;\n\t"
            "madc.hi.cc.u32   %7, %21, %25, %7;\n\t"
            "madc.hi.u32   %8, %23, %24, 0;\n\t"
            "mad.lo.cc.u32   %3, %16, %27, %3;\n\t"
            "madc.lo.cc.u32   %4, %19, %25, %4;\n\t"
            "madc.hi.cc.u32   %5, %16, %28, %5;\n\t"
            "madc.hi.cc.u32   %6, %18, %27, %6;\n\t"
            "madc.hi.cc.u32   %7, %20, %26, %7;\n\t"
            "madc.hi.cc.u32   %8, %22, %25, %8;\n\t"
            "madc.hi.u32   %9, %23, %25, 0;\n\t"
            "mad.lo.cc.u32   %4, %18, %26, %4;\n\t"
            "madc.lo.cc.u32   %5, %21, %24, %5;\n\t"
            "madc.hi.cc.u32   %6, %17, %28, %6;\n\t"
            "madc.hi.cc.u32   %7, %19, %27, %7;\n\t"
            "madc.hi.cc.u32   %8, %21, %26, %8;\n\t"
            "madc.hi.cc.u32   %9, %22, %26, %9;\n\t"
            "madc.hi.u32   %10, %23, %26, 0;\n\t"
            "mad.lo.cc.u32   %4, %17, %27, %4;\n\t"
            "madc.lo.cc.u32   %5, %20, %25, %5;\n\t"
            "madc.hi.cc.u32   %6, %16, %29, %6;\n\t"
            "madc.hi.cc.u32   %7, %18, %28, %7;\n\t"
            "madc.hi.cc.u32   %8, %20, %27, %8;\n\t"
            "madc.hi.cc.u32   %9, %21, %27, %9;\n\t"
            "madc.hi.cc.u32   %10, %22, %27, %10;\n\t"
            "madc.hi.u32   %11, %23, %27, 0;\n\t"
            "mad.lo.cc.u32   %4, %16, %28, %4;\n\t"
            "madc.lo.cc.u32   %5, %19, %26, %5;\n\t"
            "madc.lo.cc.u32   %6, %22, %24, %6;\n\t"
            "madc.hi.cc.u32   %7, %17, %29, %7;\n\t"
            "madc.hi.cc.u32   %8, %19, %28, %8;\n\t"
            "madc.hi.cc.u32   %9, %20, %28, %9;\n\t"
            "madc.hi.cc.u32   %10, %21, %28, %10;\n\t"
            "madc.hi.cc.u32   %11, %22, %28, %11;\n\t"
            "madc.hi.u32   %12, %23, %28, 0;\n\t"
            "mad.lo.cc.u32   %5, %18, %27, %5;\n\t"
            "madc.lo.cc.u32   %6, %21, %25, %6;\n\t"
            "madc.hi.cc.u32   %7, %16, %30, %7;\n\t"
            "madc.hi.cc.u32   %8, %18, %29, %8;\n\t"
            "madc.hi.cc.u32   %9, %19, %29, %9;\n\t"
            "madc.hi.cc.u32   %10, %20, %29, %10;\n\t"
            "madc.hi.cc.u32   %11, %21, %29, %11;\n\t"
            "madc.hi.cc.u32   %12, %22, %29, %12;\n\t"
            "madc.hi.u32   %13, %23, %29, 0;\n\t"
            "mad.lo.cc.u32   %5, %17, %28, %5;\n\t"
            "madc.lo.cc.u32   %6, %20, %26, %6;\n\t"
            "madc.lo.cc.u32   %7, %23, %24, %7;\n\t"
            "madc.hi.cc.u32   %8, %17, %30, %8;\n\t"
            "madc.hi.cc.u32   %9, %18, %30, %9;\n\t"
            "madc.hi.cc.u32   %10, %19, %30, %10;\n\t"
            "madc.hi.cc.u32   %11, %20, %30, %11;\n\t"
            "madc.hi.cc.u32   %12, %21, %30, %12;\n\t"
            "madc.hi.cc.u32   %13, %22, %30, %13;\n\t"
            "madc.hi.u32   %14, %23, %30, 0;\n\t"
            "mad.lo.cc.u32   %5, %16, %29, %5;\n\t"
            "madc.lo.cc.u32   %6, %19, %27, %6;\n\t"
            "madc.lo.cc.u32   %7, %22, %25, %7;\n\t"
            "madc.hi.cc.u32   %8, %16, %31, %8;\n\t"
            "madc.hi.cc.u32   %9, %17, %31, %9;\n\t"
            "madc.hi.cc.u32   %10, %18, %31, %10;\n\t"
            "madc.hi.cc.u32   %11, %19, %31, %11;\n\t"
            "madc.hi.cc.u32   %12, %20, %31, %12;\n\t"
            "madc.hi.cc.u32   %13, %21, %31, %13;\n\t"
            "madc.hi.cc.u32   %14, %22, %31, %14;\n\t"
            "madc.hi.u32   %15, %23, %31, 0;\n\t"
            "mad.lo.cc.u32   %6, %18, %28, %6;\n\t"
            "madc.lo.cc.u32   %7, %21, %26, %7;\n\t"
            "madc.lo.cc.u32   %8, %23, %25, %8;\n\t"
            "madc.lo.cc.u32   %9, %23, %26, %9;\n\t"
            "madc.lo.cc.u32   %10, %23, %27, %10;\n\t"
            "madc.lo.cc.u32   %11, %23, %28, %11;\n\t"
            "madc.lo.cc.u32   %12, %23, %29, %12;\n\t"
            "madc.lo.cc.u32   %13, %23, %30, %13;\n\t"
            "madc.lo.cc.u32   %14, %23, %31, %14;\n\t"
            "addc.cc.u32   %15, %15, 0;\n\t"
            "mad.lo.cc.u32   %6, %17, %29, %6;\n\t"
            "madc.lo.cc.u32   %7, %20, %27, %7;\n\t"
            "madc.lo.cc.u32   %8, %22, %26, %8;\n\t"
            "madc.lo.cc.u32   %9, %22, %27, %9;\n\t"
            "madc.lo.cc.u32   %10, %22, %28, %10;\n\t"
            "madc.lo.cc.u32   %11, %22, %29, %11;\n\t"
            "madc.lo.cc.u32   %12, %22, %30, %12;\n\t"
            "madc.lo.cc.u32   %13, %22, %31, %13;\n\t"
            "addc.cc.u32   %14, %14, 0;\n\t"
            "addc.cc.u32   %15, %15, 0;\n\t"
            "mad.lo.cc.u32   %6, %16, %30, %6;\n\t"
            "madc.lo.cc.u32   %7, %19, %28, %7;\n\t"
            "madc.lo.cc.u32   %8, %21, %27, %8;\n\t"
            "madc.lo.cc.u32   %9, %21, %28, %9;\n\t"
            "madc.lo.cc.u32   %10, %21, %29, %10;\n\t"
            "madc.lo.cc.u32   %11, %21, %30, %11;\n\t"
            "madc.lo.cc.u32   %12, %21, %31, %12;\n\t"
            "addc.cc.u32   %13, %13, 0;\n\t"
            "addc.cc.u32   %14, %14, 0;\n\t"
            "addc.cc.u32   %15, %15, 0;\n\t"
            "mad.lo.cc.u32   %7, %18, %29, %7;\n\t"
            "madc.lo.cc.u32   %8, %20, %28, %8;\n\t"
            "madc.lo.cc.u32   %9, %20, %29, %9;\n\t"
            "madc.lo.cc.u32   %10, %20, %30, %10;\n\t"
            "madc.lo.cc.u32   %11, %20, %31, %11;\n\t"
            "addc.cc.u32   %12, %12, 0;\n\t"
            "addc.cc.u32   %13, %13, 0;\n\t"
            "addc.cc.u32   %14, %14, 0;\n\t"
            "addc.cc.u32   %15, %15, 0;\n\t"
            "mad.lo.cc.u32   %7, %17, %30, %7;\n\t"
            "madc.lo.cc.u32   %8, %19, %29, %8;\n\t"
            "madc.lo.cc.u32   %9, %19, %30, %9;\n\t"
            "madc.lo.cc.u32   %10, %19, %31, %10;\n\t"
            "addc.cc.u32   %11, %11, 0;\n\t"
            "addc.cc.u32   %12, %12, 0;\n\t"
            "addc.cc.u32   %13, %13, 0;\n\t"
            "addc.cc.u32   %14, %14, 0;\n\t"
            "addc.cc.u32   %15, %15, 0;\n\t"
            "mad.lo.cc.u32   %7, %16, %31, %7;\n\t"
            "madc.lo.cc.u32   %8, %18, %30, %8;\n\t"
            "madc.lo.cc.u32   %9, %18, %31, %9;\n\t"
            "addc.cc.u32   %10, %10, 0;\n\t"
            "addc.cc.u32   %11, %11, 0;\n\t"
            "addc.cc.u32   %12, %12, 0;\n\t"
            "addc.cc.u32   %13, %13, 0;\n\t"
            "addc.cc.u32   %14, %14, 0;\n\t"
            "addc.cc.u32   %15, %15, 0;\n\t"
            "mad.lo.cc.u32   %8, %17, %31, %8;\n\t"
            : "=r"(w.n[0]), "=r"(w.n[1]), "=r"(w.n[2]), "=r"(w.n[3]),
                "=r"(w.n[4]), "=r"(w.n[5]), "=r"(w.n[6]), "=r"(w.n[7])
              "=r"(w.n[8]), "=r"(w.n[9]), "=r"(w.n[10]), "=r"(w.n[11]),
              "=r"(w.n[12]), "=r"(w.n[13]), "=r"(w.n[14]), "=r"(w.n[15])
            : "r"(lhs.n[0]), "r"(lhs.n[1]), "r"(lhs.n[2]), "r"(lhs.n[3]),
                "r"(lhs.n[4]), "r"(lhs.n[5]), "r"(lhs.n[6]), "r"(lhs.n[7]),
                "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]),
                "r"(rhs.n[4]), "r"(rhs.n[5]), "r"(rhs.n[6]), "r"(rhs.n[7]));

    return w;	
}

//the same logic as multiplication 128 x 128 -> 256
//but we consider limbs as 64 bit

//NB: https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/
//There are intersting considerations on 16 bit registers. May be we should use them?

DEVICE_FUNC inline uint512_g mul_uint256_to_512_asm_longregs(const uint256_g& lhs, const uint256_g& rhs)
{
    uint512_g w;
    asm (".reg .u64 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
         ".reg .u64 a0, a1, a2, a3, b0, b1, b2, b3;\n\t"
         "mov.u64         a0, %8;\n\t"
         "mov.u64         a1, %9;\n\t"
         "mov.u64         a2, %10;\n\t"
         "mov.u64         a3, %11;\n\t"
         "mov.u64         b0, %12;\n\t"
         "mov.u64         b1, %13;\n\t"
         "mov.u64         b2, %14;\n\t"
         "mov.u64         b3, %15;\n\t"
         "mul.lo.u64      r0, a0, b0;\n\t"
         "mul.hi.u64      r1, a0, b0;\n\t"
         "mad.lo.cc.u64   r1, a0, b1, r1;\n\t"
         "madc.hi.u64     r2, a0, b1, 0;\n\t"
         "mad.lo.cc.u64   r1, a1, b0, r1;\n\t"
         "madc.hi.cc.u64  r2, a1, b0, r2;\n\t"
         "madc.hi.u64     r3, a0, b2, 0;\n\t"
         "mad.lo.cc.u64   r2, a0, b2, r2;\n\t"
         "madc.hi.cc.u64  r3, a1, b1, r3;\n\t"
         "madc.hi.u64     r4, a0, b3, 0;\n\t"
         "mad.lo.cc.u64   r2, a1, b1, r2;\n\t"
         "madc.hi.cc.u64  r3, a2, b0, r3;\n\t"
         "madc.hi.cc.u64  r4, a1, b2, r4;\n\t"
         "madc.hi.u64     r5, a1, b3, 0;\n\t"
         "mad.lo.cc.u64   r2, a2, b0, r2;\n\t"
         "madc.lo.cc.u64  r3, a0, b3, r3;\n\t"
         "madc.hi.cc.u64  r4, a2, b1, r4;\n\t"
         "madc.hi.cc.u64  r5, a2, b2, r5;\n\t"
         "madc.hi.u64     r6, a2, b3, 0;\n\t"
         "mad.lo.cc.u64   r3, a1, b2, r3;\n\t"
         "madc.hi.cc.u64  r4, a3, b0, r4;\n\t"
         "madc.hi.cc.u64  r5, a3, b1, r5;\n\t"
         "madc.hi.cc.u64  r6, a3, b2, r6;\n\t"
         "madc.hi.u64     r7, a3, b3, 0;\n\t"
         "mad.lo.cc.u64   r3, a2, b1, r3;\n\t"
         "madc.lo.cc.u64  r4, a1, b3, r4;\n\t"
         "madc.lo.cc.u64  r5, a2, b3, r5;\n\t"
         "madc.lo.cc.u64  r6, a3, b3, r6;\n\t"
         "addc.u64        r7, r7, 0;\n\t"
         "mad.lo.cc.u64   r3, a3, b0, r3;\n\t"
         "madc.lo.cc.u64  r4, a2, b2, r4;\n\t"
         "madc.lo.cc.u64  r5, a3, b2, r5;\n\t"
         "addc.cc.u64     r6, r6, 0;\n\t"
         "addc.u64        r7, r7, 0;\n\t"
         "mad.lo.cc.u64   r4, a3, b1, r4;\n\t"
         "addc.cc.u64     r5, r5, 0;\n\t"
         "addc.cc.u64     r6, r6, 0;\n\t"
         "addc.u64        r7, r7, 0;\n\t"
         "mov.u64         %0, r0;\n\t"  
         "mov.u64         %1, r1;\n\t"
         "mov.u64         %2, r2;\n\t"  
         "mov.u64         %3, r3;\n\t"
         "mov.u64         %4, r4;\n\t"  
         "mov.u64         %5, r5;\n\t"
         "mov.u64         %6, r6;\n\t"  
         "mov.u64         %7, r7;\n\t"
        : "=l"(w.nn[0]), "=l"(w.nn[1]), "=l"(w.nn[2]), "=l"(w.nn[3]),
                "=l"(w.nn[4]), "=l"(w.nn[5]), "=l"(w.nn[6]), "=l"(w.nn[7])
            : "l"(lhs.nn[0]), "l"(lhs.nn[1]), "l"(lhs.nn[2]), "l"(lhs.nn[3]),
                "l"(rhs.nn[0]), "l"(rhs.nn[1]), "l"(rhs.nn[2]), "l"(rhs.nn[3]));

    return w;
}

//in order to implement Karatsuba multiplication we need addition with carry!
//HOW TO GET VALUE OF CARRY FLAG!
//NO WAY! VERY DUMB STUPID NVIDIA PTX ASSEMBLY!

struct uint128_g_caryy
{
    uint128_g res;
    uint32_t carry;
};

/*DEVICE_FUNC inline uint128_g_carry add_uint128_with_carry_asm(const uint128_g& lhs, const uint128_g& rhs)
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
}*/

//let u = u1 + u2 * 2^n
//let v = v1 + v2 * 2^n
//result = (u1 * v1) + 2^n * ((u1 + u2)(v1 + v2) - (u1 * v1)(u2 * v2))  + 2^(2n) u2 * v2;
//hence we require more space and addition operations bu only 3 multiplications (instead of four)


/*DEVICE_FUNC inline uint512_g mul_uint256_to_512_Karatsuba(const uint256_g& u, const uint256_g& v)
{
    uint256_g x = FASTEST_128_to_256_mul(u.low, v.low);
    uint256_g y = FASTEST_128_to_256_mul(u.high, v.high);
    uint256_g z = FASTEST_128_to_256_mul(add_uint128_asm(u.low, u.high), add_uint128_asm(v.low, v.high));

    //may be we should use asm for additional speed up?
	
    return w;	
}*/

#endif