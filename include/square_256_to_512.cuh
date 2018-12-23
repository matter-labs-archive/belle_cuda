#ifndef SQUARE_256_to_512_CUH
#define SQUARE_256_to_512_CUH

#include "mul_128_to_256.cuh"

//we use (The Yang–Hseih–Laih Algorithm) described in 
//https://www.sciencedirect.com/science/article/pii/S0898122109000509

DEVICE_FUNC inline uint512_g square_uint256_to_512_naive(const uint256_g& u)
{
    uint512_g w;
    #pragma unroll
	for (uint32_t j = 0; j < N_DOUBLED; j++)
        w.n[j] = 0;

    uint32_t k, temp, temp2;    
		
    #pragma unroll
	for (uint32_t i = 0; i < N; i++)
	{
        k=0;
        #pragma unroll
        for (uint32_t j = i + 1; j < N; j++)
        {
            uint32_t high_word = 0;
            uint32_t low_word = 0;
            low_word = device_long_mul(u.n[i], u.n[j], &high_word);
            low_word = device_fused_add(low_word, w.n[i + j], &high_word);
            low_word = device_fused_add(low_word, k, &high_word);
            k = high_word;
            w.n[i + j] = low_word;
        }

        w.n[N + i] = k;
    }

    k = 0;
    temp = 0;
    #pragma unroll
	for (uint32_t i = 0; i < N_DOUBLED; i++)
    {
        temp2 = w.n[i] >> 31;
        w.n[i] <<= 1;
        w.n[i] += temp;
        temp = temp2;
    }

    #pragma unroll
	for (uint32_t i = 0; i < N; i++)
    {
        uint32_t high_word = 0;
        uint32_t low_word = 0;
        low_word = device_long_mul(u.n[i], u.n[i], &high_word);
        low_word = device_fused_add(low_word, w.n[i + i], &high_word);
        low_word = device_fused_add(low_word, k, &high_word);
        w.n[i + i] = low_word;
        k = 0;
        w.n[i+i+1] = device_fused_add(w.n[i+i+1], high_word, &k);
    }

    return w;	
}

DEVICE_FUNC inline uint512_g square_uint256_to_512_asm(const uint256_g& u)
{
    uint512_g w;

     asm (  ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
            ".reg .u32 r9, r10, r11, r12, r13, r14, r15;\n\t"
            ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
#if (__CUDA_ARCH__ < 500)
            ".reg .u32 temp;\n\t"
#endif
            //unpacking operands
            "mov.b64         {a0,a1}, %8;\n\t"
            "mov.b64         {a2,a3}, %9;\n\t"
            "mov.b64         {a4,a5}, %10;\n\t"
            "mov.b64         {a6,a7}, %11;\n\t"  
             // multiplication - first stage
            "mul.lo.u32   r1, a0, a1;\n\t"           
            "mul.hi.u32   r2, a0, a1;\n\t"
            "mad.lo.cc.u32   r2, a0, a2, r2;\n\t"
            "madc.hi.u32   r3, a0, a2, 0;\n\t"
            "mad.lo.cc.u32   r3, a1, a2, r3;\n\t"
            "madc.hi.u32   r4, a1, a2, 0;\n\t"
            "mad.lo.cc.u32   r3, a0, a3, r3;\n\t"
            "madc.hi.cc.u32   r4, a0, a3, r4;\n\t"
            "madc.hi.u32   r5, a1, a3, 0;\n\t"
            "mad.lo.cc.u32   r4, a1, a3, r4;\n\t"
            "madc.hi.cc.u32   r5, a0, a4, r5;\n\t"
            "madc.hi.u32   r6, a2, a3, 0;\n\t"
            "mad.lo.cc.u32   r4, a0, a4, r4;\n\t"
            "madc.lo.cc.u32   r5, a2, a3, r5;\n\t"
            "madc.hi.cc.u32   r6, a1, a4, r6;\n\t"
            "madc.hi.u32   r7, a2, a4, 0;\n\t"
            "mad.lo.cc.u32   r5, a1, a4, r5;\n\t"
            "madc.hi.cc.u32   r6, a0, a5, r6;\n\t"
            "madc.hi.cc.u32   r7, a1, a5, r7;\n\t"
            "madc.hi.u32   r8, a3, a4, 0;\n\t"
            "mad.lo.cc.u32   r5, a0, a5, r5;\n\t"
            "madc.lo.cc.u32   r6, a2, a4, r6;\n\t"
            "madc.hi.cc.u32   r7, a0, a6, r7;\n\t"
            "madc.hi.cc.u32   r8, a2, a5, r8;\n\t"
            "madc.hi.u32   r9, a3, a5, 0;\n\t"
            "mad.lo.cc.u32   r6, a1, a5, r6;\n\t"
            "madc.lo.cc.u32   r7, a3, a4, r7;\n\t"
            "madc.hi.cc.u32   r8, a1, a6, r8;\n\t"
            "madc.hi.cc.u32   r9, a2, a6, r9;\n\t"
            "madc.hi.u32   r10, a4, a5, 0;\n\t"
            "mad.lo.cc.u32   r6, a0, a6, r6;\n\t"
            "madc.lo.cc.u32   r7, a2, a5, r7;\n\t"
            "madc.hi.cc.u32   r8, a0, a7, r8;\n\t"
            "madc.hi.cc.u32   r9, a1, a7, r9;\n\t"
            "madc.hi.cc.u32   r10, a3, a6, r10;\n\t"
            "madc.hi.u32   r11, a4, a6, 0;\n\t"
            "mad.lo.cc.u32   r7, a1, a6, r7;\n\t"
            "madc.lo.cc.u32   r8, a3, a5, r8;\n\t"
            "madc.lo.cc.u32   r9, a4, a5, r9;\n\t"
            "madc.hi.cc.u32   r10, a2, a7, r10;\n\t"
            "madc.hi.cc.u32   r11, a3, a7, r11;\n\t"
            "madc.hi.u32   r12, a5, a6, 0;\n\t"
            "mad.lo.cc.u32   r7, a0, a7, r7;\n\t"
            "madc.lo.cc.u32   r8, a2, a6, r8;\n\t"
            "madc.lo.cc.u32   r9, a3, a6, r9;\n\t"
            "madc.lo.cc.u32   r10, a4, a6, r10;\n\t"
            "madc.lo.cc.u32   r11, a5, a6, r11;\n\t"
            "madc.hi.cc.u32   r12, a4, a7, r12;\n\t"
            "madc.hi.u32   r13, a5, a7, 0;\n\t"
            "mad.lo.cc.u32   r8, a1, a7, r8;\n\t"
            "madc.lo.cc.u32   r9, a2, a7, r9;\n\t"
            "madc.lo.cc.u32   r10, a3, a7, r10;\n\t"
            "madc.lo.cc.u32   r11, a4, a7, r11;\n\t"
            "madc.lo.cc.u32   r12, a5, a7, r12;\n\t"
            "madc.lo.cc.u32   r13, a6, a7, r13;\n\t"
            "madc.hi.u32   r14, a6, a7, 0;\n\t"
            //shifting
#if (__CUDA_ARCH__ >= 500)
            "shf.l.clamp.b32 r15, r14, r15, 1;\n\t"
            "shf.l.clamp.b32 r14, r13, r14, 1;\n\t"
            "shf.l.clamp.b32 r13, r12, r13, 1;\n\t"
            "shf.l.clamp.b32 r12, r11, r12, 1;\n\t"
            "shf.l.clamp.b32 r11, r10, r11, 1;\n\t"
            "shf.l.clamp.b32 r10, r9, r10, 1;\n\t"
            "shf.l.clamp.b32 r9,  r8, r9, 1;\n\t"
            "shf.l.clamp.b32 r8,  r7, r8, 1;\n\t"
            "shf.l.clamp.b32 r7,  r6, r7, 1;\n\t"
            "shf.l.clamp.b32 r6, r5, r6, 1;\n\t"
            "shf.l.clamp.b32 r5, r4, r5, 1;\n\t"
            "shf.l.clamp.b32 r4, r3, r4, 1;\n\t"
            "shf.l.clamp.b32 r3, r2, r3, 1;\n\t"
            "shf.l.clamp.b32 r2, r1, r2, 1;\n\t"
            "shf.l.clamp.b32 r1, r0, r1, 1;\n\t"
            "shl.b32 r0, r0, 1;\n\t"
#else
            "shr.b32 r15, r14, 31;\n\t"
            "shl.b32 r14, r14, 1;\n\t"
            "shr.b32 temp, r13, 31;\n\t"
            "or.b32 r14, r14, temp;\n\t"
            "shl.b32 r13, r13, 1;\n\t"
            "shr.b32 temp, r12, 31;\n\t"
            "or.b32 r13, r13, temp;\n\t"
            "shl.b32 r12, r12, 1;\n\t"
            "shr.b32 temp, r11, 31;\n\t"
            "or.b32 r12, r12, temp;\n\t"
            "shl.b32 r11, r11, 1;\n\t"
            "shr.b32 temp, r10, 31;\n\t"
            "or.b32 r11, r11, temp;\n\t"
            "shl.b32 r10, r10, 1;\n\t"
            "shr.b32 temp, r9, 31;\n\t"
            "or.b32 r10, r10, temp;\n\t"
            "shl.b32 r9, r9, 1;\n\t"
            "shr.b32 temp, r8, 31;\n\t"
            "or.b32 r9, r9, temp;\n\t"
            "shl.b32 r8, r8, 1;\n\t"
            "shr.b32 temp, r7, 31;\n\t"
            "or.b32 r8, r8, temp;\n\t"
            "shl.b32 r7, r7, 1;\n\t"
            "shr.b32 temp, r6, 31;\n\t"
            "or.b32 r7, r7, temp;\n\t"
            "shl.b32 r6, r6, 1;\n\t"
            "shr.b32 temp, r5, 31;\n\t"
            "or.b32 r6, r6, temp;\n\t"
            "shl.b32 r5, r5, 1;\n\t"
            "shr.b32 temp, r4, 31;\n\t"
            "or.b32 r5, r5, temp;\n\t"
            "shl.b32 r4, r4, 1;\n\t"
            "shr.b32 temp, r3, 31;\n\t"
            "or.b32 r4, r4, temp;\n\t"
            "shl.b32 r3, r3, 1;\n\t"
            "shr.b32 temp, r2, 31;\n\t"
            "or.b32 r3, r3, temp;\n\t"
            "shl.b32 r2, r2, 1;\n\t"
            "shr.b32 temp, r1, 31;\n\t"
            "or.b32 r2, r2, temp;\n\t"
            "shl.b32 r1, r1, 1;\n\t"
            "shr.b32 temp, r0, 31;\n\t"
            "or.b32 r1, r1, temp;\n\t"
            "shl.b32 r0, r0, 1;\n\t"
#endif
            //final multiplication
            "mad.lo.cc.u32 r0, a0, a0, r0;\n\t"
            "madc.hi.cc.u32 r1, a0, a0, r1;\n\t"
            "madc.lo.cc.u32 r2, a1, a1, r2;\n\t"
            "madc.hi.cc.u32 r3, a1, a1, r3;\n\t"
            "madc.lo.cc.u32 r4, a2, a2, r4;\n\t"
            "madc.hi.cc.u32 r5, a2, a2, r5;\n\t"
            "madc.lo.cc.u32 r6, a3, a3, r6;\n\t"
            "madc.hi.cc.u32 r7, a3, a3, r7;\n\t"
            "madc.lo.cc.u32 r8, a4, a4, r8;\n\t"
            "madc.hi.cc.u32 r9, a4, a4, r9;\n\t"
            "madc.lo.cc.u32 r10, a5, a5, r10;\n\t"
            "madc.hi.cc.u32 r11, a5, a5, r11;\n\t"
            "madc.lo.cc.u32 r12, a6, a6, r12;\n\t"
            "madc.hi.cc.u32 r13, a6, a6, r13;\n\t"
            "madc.lo.cc.u32 r14, a7, a7, r14;\n\t"
            "madc.hi.cc.u32 r15, a7, a7, r15;\n\t"
            //packing result
            "mov.b64         %0, {r0,r1};\n\t"  
            "mov.b64         %1, {r2,r3};\n\t"
            "mov.b64         %2, {r4,r5};\n\t"  
            "mov.b64         %3, {r6,r7};\n\t"
            "mov.b64         %4, {r8,r9};\n\t"  
            "mov.b64         %5, {r10,r11};\n\t"
            "mov.b64         %6, {r12,r13};\n\t"  
            "mov.b64         %7, {r14,r15};\n\t"
            : "=l"(w.nn[0]), "=l"(w.nn[1]), "=l"(w.nn[2]), "=l"(w.nn[3]),
                "=l"(w.nn[4]), "=l"(w.nn[5]), "=l"(w.nn[6]), "=l"(w.nn[7])
            : "l"(u.nn[0]), "l"(u.nn[1]), "l"(u.nn[2]), "l"(u.nn[3]));

    return w;
}


#endif
