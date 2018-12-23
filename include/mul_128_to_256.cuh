#ifndef MUL_128_to_256_CUH
#define MUL_128_to_256_CUH

#include "cuda_structs.cuh"

//helper functions for naive multiplication & naive mult itself
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------


DEVICE_FUNC inline uint32_t device_long_mul(uint32_t x, uint32_t y, uint32_t* high_ptr)
	{
		uint32_t high = __umulhi(x, y);
		*high_ptr = high;
		return x * y;
	}

DEVICE_FUNC inline uint32_t device_fused_add(uint32_t x, uint32_t y, uint32_t* high_ptr)
{
	uint32_t z = x + y;
	if (z < x)
		(*high_ptr)++;
    return z;
}	

DEVICE_FUNC inline uint256_g mul_uint128_to_256_naive(const uint128_g& u, const uint128_g& v)
{
    uint256_g w;
		
    #pragma unroll
	for (uint32_t j = 0; j < HALF_N; j++)
	{
        uint32_t k = 0;

        #pragma unroll
        for (uint32_t i = 0; i < HALF_N; i++)
        {
            uint32_t high_word = 0;
            uint32_t low_word = 0;
            low_word = device_long_mul(u.n[i], v.n[j], &high_word);
            low_word = device_fused_add(low_word, w.n[i + j], &high_word);
            low_word = device_fused_add(low_word, k, &high_word);
            k = high_word;
            w.n[i + j] = low_word;
        }

        w.n[HALF_N + j] = k;
    }

    return w;	
}

//asm based multi[lications:
//NB: the only difference between ver1 and ver2 is additional register allocation
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

//this code was produced by my generator
DEVICE_FUNC inline uint256_g mul_uint128_to_256_asm_ver1(const uint128_g& lhs, const uint128_g& rhs)
{
    uint256_g w;
    asm (       "mul.lo.u32   %0, %8, %12;\n\t"
                "mul.hi.u32   %1, %8, %12;\n\t"
                "mad.lo.cc.u32   %1, %9, %12, %1;\n\t"
                "madc.hi.u32   %2, %9, %12, 0;\n\t"
                "mad.lo.cc.u32   %1, %8, %13, %1;\n\t"
                "madc.hi.cc.u32   %2, %8, %13, %2;\n\t"
                "madc.hi.u32   %3, %10, %12, 0;\n\t"
                "mad.lo.cc.u32   %2, %10, %12, %2;\n\t"
                "madc.hi.cc.u32   %3, %9, %13, %3;\n\t"
                "madc.hi.u32   %4, %11, %12, 0;\n\t"
                "mad.lo.cc.u32   %2, %9, %13, %2;\n\t"
                "madc.hi.cc.u32   %3, %8, %14, %3;\n\t"
                "madc.hi.cc.u32   %4, %10, %13, %4;\n\t"
                "madc.hi.u32   %5, %11, %13, 0;\n\t"
                "mad.lo.cc.u32   %2, %8, %14, %2;\n\t"
                "madc.lo.cc.u32   %3, %11, %12, %3;\n\t"
                "madc.hi.cc.u32   %4, %9, %14, %4;\n\t"
                "madc.hi.cc.u32   %5, %10, %14, %5;\n\t"
                "madc.hi.u32   %6, %11, %14, 0;\n\t"
                "mad.lo.cc.u32   %3, %10, %13, %3;\n\t"
                "madc.hi.cc.u32   %4, %8, %15, %4;\n\t"
                "madc.hi.cc.u32   %5, %9, %15, %5;\n\t"
                "madc.hi.cc.u32   %6, %10, %15, %6;\n\t"
                "madc.hi.u32   %7, %11, %15, 0;\n\t"
                "mad.lo.cc.u32   %3, %9, %14, %3;\n\t"
                "madc.lo.cc.u32   %4, %11, %13, %4;\n\t"
                "madc.lo.cc.u32   %5, %11, %14, %5;\n\t"
                "madc.lo.cc.u32   %6, %11, %15, %6;\n\t"
                "addc.cc.u32   %7, %7, 0;\n\t"
                "mad.lo.cc.u32   %3, %8, %15, %3;\n\t"
                "madc.lo.cc.u32   %4, %10, %14, %4;\n\t"
                "madc.lo.cc.u32   %5, %10, %15, %5;\n\t"
                "addc.cc.u32   %6, %6, 0;\n\t"
                "addc.cc.u32   %7, %7, 0;\n\t"
                "mad.lo.cc.u32   %4, %9, %15, %4;\n\t"
                : "=r"(w.n[0]), "=r"(w.n[1]), "=r"(w.n[2]), "=r"(w.n[3]),
                    "=r"(w.n[4]), "=r"(w.n[5]), "=r"(w.n[6]), "=r"(w.n[7])
                : "r"(lhs.n[0]), "r"(lhs.n[1]), "r"(lhs.n[2]), "r"(lhs.n[3]),
                    "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]));
    return w;	
}

//the following two samples of optimized asm multiplication code is taken from:
//https://devtalk.nvidia.com/default/topic/1017754/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/


// multiply two unsigned 128-bit integers into an unsigned 256-bit product
DEVICE_FUNC inline uint256_g mul_uint128_to_256_asm_ver2(const uint128_g& a, const uint128_g& b)
{
    uint256_g res;
    asm ("{\n\t"
         ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
         ".reg .u32 a0, a1, a2, a3, b0, b1, b2, b3;\n\t"
         "mov.b64         {a0,a1}, %4;\n\t"
         "mov.b64         {a2,a3}, %5;\n\t"
         "mov.b64         {b0,b1}, %6;\n\t"
         "mov.b64         {b2,b3}, %7;\n\t"
         "mul.lo.u32      r0, a0, b0;\n\t"
         "mul.hi.u32      r1, a0, b0;\n\t"
         "mad.lo.cc.u32   r1, a0, b1, r1;\n\t"
         "madc.hi.u32     r2, a0, b1, 0;\n\t"
         "mad.lo.cc.u32   r1, a1, b0, r1;\n\t"
         "madc.hi.cc.u32  r2, a1, b0, r2;\n\t"
         "madc.hi.u32     r3, a0, b2, 0;\n\t"
         "mad.lo.cc.u32   r2, a0, b2, r2;\n\t"
         "madc.hi.cc.u32  r3, a1, b1, r3;\n\t"
         "madc.hi.u32     r4, a0, b3, 0;\n\t"
         "mad.lo.cc.u32   r2, a1, b1, r2;\n\t"
         "madc.hi.cc.u32  r3, a2, b0, r3;\n\t"
         "madc.hi.cc.u32  r4, a1, b2, r4;\n\t"
         "madc.hi.u32     r5, a1, b3, 0;\n\t"
         "mad.lo.cc.u32   r2, a2, b0, r2;\n\t"
         "madc.lo.cc.u32  r3, a0, b3, r3;\n\t"
         "madc.hi.cc.u32  r4, a2, b1, r4;\n\t"
         "madc.hi.cc.u32  r5, a2, b2, r5;\n\t"
         "madc.hi.u32     r6, a2, b3, 0;\n\t"
         "mad.lo.cc.u32   r3, a1, b2, r3;\n\t"
         "madc.hi.cc.u32  r4, a3, b0, r4;\n\t"
         "madc.hi.cc.u32  r5, a3, b1, r5;\n\t"
         "madc.hi.cc.u32  r6, a3, b2, r6;\n\t"
         "madc.hi.u32     r7, a3, b3, 0;\n\t"
         "mad.lo.cc.u32   r3, a2, b1, r3;\n\t"
         "madc.lo.cc.u32  r4, a1, b3, r4;\n\t"
         "madc.lo.cc.u32  r5, a2, b3, r5;\n\t"
         "madc.lo.cc.u32  r6, a3, b3, r6;\n\t"
         "addc.u32        r7, r7, 0;\n\t"
         "mad.lo.cc.u32   r3, a3, b0, r3;\n\t"
         "madc.lo.cc.u32  r4, a2, b2, r4;\n\t"
         "madc.lo.cc.u32  r5, a3, b2, r5;\n\t"
         "addc.cc.u32     r6, r6, 0;\n\t"
         "addc.u32        r7, r7, 0;\n\t"
         "mad.lo.cc.u32   r4, a3, b1, r4;\n\t"
         "addc.cc.u32     r5, r5, 0;\n\t"
         "addc.cc.u32     r6, r6, 0;\n\t"
         "addc.u32        r7, r7, 0;\n\t"
         "mov.b64         %0, {r0,r1};\n\t"  
         "mov.b64         %1, {r2,r3};\n\t"
         "mov.b64         %2, {r4,r5};\n\t"  
         "mov.b64         %3, {r6,r7};\n\t"
         "}"
         : "=l"(res.nn[0]), "=l"(res.nn[1]), "=l"(res.nn[2]), "=l"(res.nn[3])
         : "l"(a.low), "l"(a.high), "l"(b.low), "l"(b.high));

    return res;
}

//NB: I do not have enough CUDA capabilities to benchmark this implementation!

#if (__CUDA_ARCH__ >= 500)
DEVICE_FUNC inline uint256_g mul_uint128_to_256_asm_ver3(const uint128_g& a, const uint128_g& b)
{
    uint256_g res;
    asm ("{\n\t"
         ".reg .u32 aa0, aa1, aa2, aa3, bb0, bb1, bb2, bb3;\n\t"
         ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
         ".reg .u32 s0, s1, s2, s3, s4, s5, s6, s7;\n\t"
         ".reg .u32 t0, t1, t2, t3, t4, t5, t6, t7;\n\t"
         ".reg .u16 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
         ".reg .u16 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
         // unpack source operands
         "mov.b64         {aa0,aa1}, %4;\n\t"
         "mov.b64         {aa2,aa3}, %5;\n\t"
         "mov.b64         {bb0,bb1}, %6;\n\t"
         "mov.b64         {bb2,bb3}, %7;\n\t"
         "mov.b32         {a0,a1}, aa0;\n\t"
         "mov.b32         {a2,a3}, aa1;\n\t"
         "mov.b32         {a4,a5}, aa2;\n\t"
         "mov.b32         {a6,a7}, aa3;\n\t"
         "mov.b32         {b0,b1}, bb0;\n\t"
         "mov.b32         {b2,b3}, bb1;\n\t"
         "mov.b32         {b4,b5}, bb2;\n\t"
         "mov.b32         {b6,b7}, bb3;\n\t"
         // compute first partial sum
         "mul.wide.u16    r0, a0, b0;\n\t"
         "mul.wide.u16    r1, a0, b2;\n\t"
         "mul.wide.u16    r2, a0, b4;\n\t"
         "mul.wide.u16    r3, a0, b6;\n\t"
         "mul.wide.u16    r4, a1, b7;\n\t"
         "mul.wide.u16    r5, a3, b7;\n\t"
         "mul.wide.u16    r6, a5, b7;\n\t"
         "mul.wide.u16    r7, a7, b7;\n\t"
         "mul.wide.u16    t3, a1, b5;\n\t"
         "mul.wide.u16    t4, a2, b6;\n\t"
         "add.cc.u32      r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.u32        r5, r5, 0;\n\t"
         "mul.wide.u16    t3, a2, b4;\n\t"
         "mul.wide.u16    t4, a3, b5;\n\t"
         "add.cc.u32      r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.u32        r5, r5, 0;\n\t"
         "mul.wide.u16    t2, a1, b3;\n\t"
         "mul.wide.u16    t3, a3, b3;\n\t"
         "mul.wide.u16    t4, a4, b4;\n\t"
         "mul.wide.u16    t5, a4, b6;\n\t"
         "add.cc.u32      r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.u32        r6, r6, 0;\n\t"
         "mul.wide.u16    t2, a2, b2;\n\t"
         "mul.wide.u16    t3, a4, b2;\n\t"
         "mul.wide.u16    t4, a5, b3;\n\t"
         "mul.wide.u16    t5, a5, b5;\n\t"
         "add.cc.u32      r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.u32        r6, r6, 0;\n\t"
         "mul.wide.u16    t1, a1, b1;\n\t"
         "mul.wide.u16    t2, a3, b1;\n\t"
         "mul.wide.u16    t3, a5, b1;\n\t"
         "mul.wide.u16    t4, a6, b2;\n\t"
         "mul.wide.u16    t5, a6, b4;\n\t"
         "mul.wide.u16    t6, a6, b6;\n\t"
         "add.cc.u32      r1, r1, t1;\n\t"
         "addc.cc.u32     r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.cc.u32     r6, r6, t6;\n\t"
         "addc.u32        r7, r7, 0;\n\t"
         "mul.wide.u16    t1, a2, b0;\n\t"
         "mul.wide.u16    t2, a4, b0;\n\t"
         "mul.wide.u16    t3, a6, b0;\n\t"
         "mul.wide.u16    t4, a7, b1;\n\t"
         "mul.wide.u16    t5, a7, b3;\n\t"
         "mul.wide.u16    t6, a7, b5;\n\t"
         "add.cc.u32      r1, r1, t1;\n\t"
         "addc.cc.u32     r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.cc.u32     r6, r6, t6;\n\t"
         "addc.u32        r7, r7, 0;\n\t"
         // compute second partial sum
         "mul.wide.u16    t0, a0, b1;\n\t"
         "mul.wide.u16    t1, a0, b3;\n\t"
         "mul.wide.u16    t2, a0, b5;\n\t"
         "mul.wide.u16    t3, a0, b7;\n\t"
         "mul.wide.u16    t4, a2, b7;\n\t"
         "mul.wide.u16    t5, a4, b7;\n\t"
         "mul.wide.u16    t6, a6, b7;\n\t"
         "mul.wide.u16    s3, a1, b6;\n\t"
         "add.cc.u32      t3, t3, s3;\n\t"
         "addc.u32        t4, t4, 0;\n\t"
         "mul.wide.u16    s3, a2, b5;\n\t"
         "add.cc.u32      t3, t3, s3;\n\t"
         "addc.u32        t4, t4, 0;\n\t"
         "mul.wide.u16    s2, a1, b4;\n\t"
         "mul.wide.u16    s3, a3, b4;\n\t"
         "mul.wide.u16    s4, a3, b6;\n\t"
         "add.cc.u32      t2, t2, s2;\n\t"
         "addc.cc.u32     t3, t3, s3;\n\t"
         "addc.cc.u32     t4, t4, s4;\n\t"
         "addc.u32        t5, t5, 0;\n\t"
         "mul.wide.u16    s2, a2, b3;\n\t"
         "mul.wide.u16    s3, a4, b3;\n\t"
         "mul.wide.u16    s4, a4, b5;\n\t"
         "add.cc.u32      t2, t2, s2;\n\t"
         "addc.cc.u32     t3, t3, s3;\n\t"
         "addc.cc.u32     t4, t4, s4;\n\t"
         "addc.u32        t5, t5, 0;\n\t"
         "mul.wide.u16    s1, a1, b2;\n\t"
         "mul.wide.u16    s2, a3, b2;\n\t"
         "mul.wide.u16    s3, a5, b2;\n\t"
         "mul.wide.u16    s4, a5, b4;\n\t"
         "mul.wide.u16    s5, a5, b6;\n\t"
         "add.cc.u32      t1, t1, s1;\n\t"
         "addc.cc.u32     t2, t2, s2;\n\t"
         "addc.cc.u32     t3, t3, s3;\n\t"
         "addc.cc.u32     t4, t4, s4;\n\t"
         "addc.cc.u32     t5, t5, s5;\n\t"
         "addc.u32        t6, t6, 0;\n\t"
         "mul.wide.u16    s1, a2, b1;\n\t"
         "mul.wide.u16    s2, a4, b1;\n\t"
         "mul.wide.u16    s3, a6, b1;\n\t"
         "mul.wide.u16    s4, a6, b3;\n\t"
         "mul.wide.u16    s5, a6, b5;\n\t"
         "add.cc.u32      t1, t1, s1;\n\t"
         "addc.cc.u32     t2, t2, s2;\n\t"
         "addc.cc.u32     t3, t3, s3;\n\t"
         "addc.cc.u32     t4, t4, s4;\n\t"
         "addc.cc.u32     t5, t5, s5;\n\t"
         "addc.u32        t6, t6, 0;\n\t"
         "mul.wide.u16    s0, a1, b0;\n\t"
         "mul.wide.u16    s1, a3, b0;\n\t"
         "mul.wide.u16    s2, a5, b0;\n\t"
         "mul.wide.u16    s3, a7, b0;\n\t"
         "mul.wide.u16    s4, a7, b2;\n\t"
         "mul.wide.u16    s5, a7, b4;\n\t"
         "mul.wide.u16    s6, a7, b6;\n\t"
         "add.cc.u32      t0, t0, s0;\n\t"
         "addc.cc.u32     t1, t1, s1;\n\t"
         "addc.cc.u32     t2, t2, s2;\n\t"
         "addc.cc.u32     t3, t3, s3;\n\t"
         "addc.cc.u32     t4, t4, s4;\n\t"
         "addc.cc.u32     t5, t5, s5;\n\t"
         "addc.cc.u32     t6, t6, s6;\n\t"
         "addc.u32        t7, 0, 0;\n\t"
         // offset second partial sum by 16 bits
         "shf.l.clamp.b32 s7, t6, t7, 16;\n\t"
         "shf.l.clamp.b32 s6, t5, t6, 16;\n\t"
         "shf.l.clamp.b32 s5, t4, t5, 16;\n\t"
         "shf.l.clamp.b32 s4, t3, t4, 16;\n\t"
         "shf.l.clamp.b32 s3, t2, t3, 16;\n\t"
         "shf.l.clamp.b32 s2, t1, t2, 16;\n\t"
         "shf.l.clamp.b32 s1, t0, t1, 16;\n\t"
         "shf.l.clamp.b32 s0,  0, t0, 16;\n\t"
         // add partial sums
         "add.cc.u32      r0, r0, s0;\n\t"
         "addc.cc.u32     r1, r1, s1;\n\t"
         "addc.cc.u32     r2, r2, s2;\n\t"
         "addc.cc.u32     r3, r3, s3;\n\t"
         "addc.cc.u32     r4, r4, s4;\n\t"
         "addc.cc.u32     r5, r5, s5;\n\t"
         "addc.cc.u32     r6, r6, s6;\n\t"
         "addc.u32        r7, r7, s7;\n\t"
         // pack up result
         "mov.b64         %0, {r0,r1};\n\t"  
         "mov.b64         %1, {r2,r3};\n\t"
         "mov.b64         %2, {r4,r5};\n\t"  
         "mov.b64         %3, {r6,r7};\n\t"
         "}"
         : "=l"(res.nn[0]), "=l"(res.nn[1]), "=l"(res.nn[2]), "=l"(res.nn[3])
         : "l"(a.low), "l"(a.high), "l"(b.low), "l"(b.high));

    return res;
}
#endif

#define FASTEST_128_to_256_mul(a, b) mul_uint128_to_256_asm_ver1(a, b)

#endif