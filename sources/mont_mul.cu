#include "cuda_structs.h"

//multiplication in Montgomery form
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC uint256_g mont_mul_256_naive_SOS(const uint256_g& u, const uint256_g& v)
{
    uint512_g T = MUL(u, v);
    uint256_g res;
	
    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t carry = 0;
        uint32_t m = T.n[i] * BASE_FIELD_N;

        #pragma unroll
        for (uint32_t j = 0; j < N; j++)
        {
            uint32_t high_word = 0;
            uint32_t low_word = device_long_mul(m, BASE_FIELD_P.n[j], &high_word);
            low_word = device_fused_add(low_word, T.n[i + j], &high_word);
            low_word = device_fused_add(low_word, carry, &high_word);

            T.n[i + j] = low_word;
            carry = high_word;
        }
        //continue carrying
        uint32_t j = N;
        while (carry)
        {
            uint32_t new_carry = 0;
            T.n[i + j] = device_fused_add(T.n[i + j], carry, &new_carry);
            j++;
            carry = new_carry;
        }
    }
    
    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
    {
        res.n[i] = T.n[i + N];
    }

    if (CMP(res, BASE_FIELD_P) >= 0)
    {
        //TODO: may be better change to unary version of sub?
        res = SUB(res, BASE_FIELD_P);
    }

    return res;		
}

DEVICE_FUNC uint256_g mont_mul_256_naive_CIOS(const uint256_g& u, const uint256_g& v)
{
    uint256_g T;

    #pragma unroll
	for (uint32_t j = 0; j < N; j++)
        T.n[j] = 0;

    uint32_t prefix_low = 0, prefix_high = 0, m;
    uint32_t high_word, low_word;

    #pragma unroll
    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t carry = 0;
        #pragma unroll
        for (uint32_t j = 0; j < N; j++)
        {         
            low_word = device_long_mul(u.n[j], v.n[i], &high_word);
            low_word = device_fused_add(low_word, T.n[j], &high_word);
            low_word = device_fused_add(low_word, carry, &high_word);
            carry = high_word;
            T.n[j] = low_word;
        }

        //TODO: may be we actually require less space? (only one additional limb instead of two)
        prefix_high = 0;
        prefix_low = device_fused_add(prefix_low, carry, &prefix_high);

        m = T.n[0] * BASE_FIELD_N;
        low_word = device_long_mul(BASE_FIELD_P.n[0], m, &high_word);
        low_word = device_fused_add(low_word, T.n[0], &high_word);
        carry = high_word;

        #pragma unroll
        for (uint32_t j = 1; j < N; j++)
        {
            low_word = device_long_mul(BASE_FIELD_P.n[j], m, &high_word);
            low_word = device_fused_add(low_word, T.n[j], &high_word);
            low_word = device_fused_add(low_word, carry, &high_word);
            T.n[j-1] = low_word;
            carry = high_word;
        }

        T.n[N-1] = device_fused_add(prefix_low, carry, &prefix_high);
        prefix_low = prefix_high;
    }
    
    if (CMP(T, BASE_FIELD_P) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        T = SUB(T, BASE_FIELD_P);
    }

    return T;
}

DEVICE_FUNC uint256_g mont_mul_256_asm_SOS(const uint256_g& u, const uint256_g& v)
{
    uint512_g T = MUL(u, v);
    uint256_g w;

    asm (   ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7, a8;\n\t"
            ".reg .u32 a9, a10, a11, a12, a13, a14, a15;\n\t"
            ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"
            ".reg .u32 m, q, carry;\n\t"
            //unpacking operands
            "mov.b64         {a0,a1}, %4;\n\t"
            "mov.b64         {a2,a3}, %5;\n\t"
            "mov.b64         {a4,a5}, %6;\n\t"
            "mov.b64         {a6,a7}, %7;\n\t"
            "mov.b64         {a8,a9}, %8;\n\t"
            "mov.b64         {a10,a11}, %9;\n\t"
            "mov.b64         {a12,a13}, %10;\n\t"
            "mov.b64         {a14,a15}, %11;\n\t"
            "ld.const.u32    n0, [BASE_FIELD_P];\n\t"
            "ld.const.u32    n1, [BASE_FIELD_P + 4];\n\t"
            "ld.const.u32    n2, [BASE_FIELD_P + 8];\n\t"
            "ld.const.u32    n3, [BASE_FIELD_P + 12];\n\t"
            "ld.const.u32    n4, [BASE_FIELD_P + 16];\n\t"
            "ld.const.u32    n5, [BASE_FIELD_P + 20];\n\t"
            "ld.const.u32    n6, [BASE_FIELD_P + 24];\n\t"
            "ld.const.u32    n7, [BASE_FIELD_P + 28];\n\t"
            "ld.const.u32    q, [BASE_FIELD_N];\n\t"
            //main routine
            "mul.lo.u32   m, a0, q;\n\t"
            "mad.lo.cc.u32  a0, m, n0, a0;\n\t"
            "madc.lo.cc.u32  a1, m, n1, a1;\n\t"
            "madc.lo.cc.u32  a2, m, n2, a2;\n\t"
            "madc.lo.cc.u32  a3, m, n3, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n4, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n5, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n6, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n7, a7;\n\t"
            "addc.cc.u32  a8, a8, 0;\n\t"
            "addc.u32 carry, 0, 0;\n\t"
            
            "mad.hi.cc.u32  a1, m, n0, a1;\n\t"
            "madc.hi.cc.u32  a2, m, n1, a2;\n\t"
            "madc.hi.cc.u32  a3, m, n2, a3;\n\t"
            "madc.hi.cc.u32  a4, m, n3, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n4, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n5, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n6, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n7, a8;\n\t"
            "addc.cc.u32  a9, a9, carry;\n\t"
            "addc.u32 carry, 0, 0;\n\t"
            
            "mul.lo.u32   m, a1, q;\n\t"
            "mad.lo.cc.u32  a1, m, n0, a1;\n\t"
            "madc.lo.cc.u32  a2, m, n1, a2;\n\t"
            "madc.lo.cc.u32  a3, m, n2, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n3, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n4, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n5, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n6, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n7, a8;\n\t"
            "addc.cc.u32  a9, a9, 0;\n\t"
            "addc.u32 carry, carry, 0;\n\t"
           
            "mad.hi.cc.u32  a2, m, n0, a2;\n\t"
            "madc.hi.cc.u32  a3, m, n1, a3;\n\t"
            "madc.hi.cc.u32  a4, m, n2, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n3, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n4, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n5, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n6, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n7, a9;\n\t"
            "addc.cc.u32  a10, a10, carry;\n\t"
            "addc.u32 carry, 0, 0;\n\t"
            
            "mul.lo.u32   m, a2, q;\n\t"
            "mad.lo.cc.u32  a2, m, n0, a2;\n\t"
            "madc.lo.cc.u32  a3, m, n1, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n2, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n3, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n4, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n5, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n6, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n7, a9;\n\t"
            "addc.cc.u32  a10, a10, 0;\n\t"
            "addc.u32 carry, carry, 0;\n\t"
            
            "mad.hi.cc.u32  a3, m, n0, a3;\n\t"
            "madc.hi.cc.u32  a4, m, n1, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n2, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n3, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n4, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n5, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n6, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n7, a10;\n\t"
            "addc.cc.u32  a11, a11, carry;\n\t"
            "addc.u32 carry, 0, 0;\n\t"
            
            "mul.lo.u32   m, a3, q;\n\t"
            "mad.lo.cc.u32  a3, m, n0, a3;\n\t"
            "madc.lo.cc.u32  a4, m, n1, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n2, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n3, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n4, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n5, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n6, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n7, a10;\n\t"
            "addc.cc.u32  a11, a11, 0;\n\t"
            "addc.u32 carry, carry, 0;\n\t"
            
            "mad.hi.cc.u32  a4, m, n0, a4;\n\t"
            "madc.hi.cc.u32  a5, m, n1, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n2, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n3, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n4, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n5, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n6, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n7, a11;\n\t"
            "addc.cc.u32  a12, a12, carry;\n\t"
            "addc.u32 carry, 0, 0;\n\t"

            "mul.lo.u32   m, a4, q;\n\t"
            "mad.lo.cc.u32  a4, m, n0, a4;\n\t"
            "madc.lo.cc.u32  a5, m, n1, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n2, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n3, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n4, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n5, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n6, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n7, a11;\n\t"
            "addc.cc.u32  a12, a12, 0;\n\t"
            "addc.u32 carry, carry, 0;\n\t"

            "mad.hi.cc.u32  a5, m, n0, a5;\n\t"
            "madc.hi.cc.u32  a6, m, n1, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n2, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n3, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n4, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n5, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n6, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n7, a12;\n\t"
            "addc.cc.u32  a13, a13, carry;\n\t"
            "addc.u32 carry, 0, 0;\n\t"
           
            "mul.lo.u32   m, a5, q;\n\t"
            "mad.lo.cc.u32  a5, m, n0, a5;\n\t"
            "madc.lo.cc.u32  a6, m, n1, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n2, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n3, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n4, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n5, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n6, a11;\n\t"
            "madc.lo.cc.u32  a12, m, n7, a12;\n\t"
            "addc.cc.u32  a13, a13, 0;\n\t"
            "addc.u32 carry, carry, 0;\n\t"
            
            "mad.hi.cc.u32  a6, m, n0, a6;\n\t"
            "madc.hi.cc.u32  a7, m, n1, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n2, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n3, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n4, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n5, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n6, a12;\n\t"
            "madc.hi.cc.u32  a13, m, n7, a13;\n\t"
            "addc.cc.u32  a14, a14, carry;\n\t"
            "addc.u32  a15, a15, 0;\n\t"

            "mul.lo.u32   m, a6, q;\n\t"
            "mad.lo.cc.u32  a6, m, n0, a6;\n\t"
            "madc.lo.cc.u32  a7, m, n1, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n2, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n3, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n4, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n5, a11;\n\t"
            "madc.lo.cc.u32  a12, m, n6, a12;\n\t"
            "madc.lo.cc.u32  a13, m, n7, a13;\n\t"
            "addc.cc.u32  a14, a14, 0;\n\t"
            "addc.u32  a15, a15, 0;\n\t"

            "mad.hi.cc.u32  a7, m, n0, a7;\n\t"
            "madc.hi.cc.u32  a8, m, n1, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n2, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n3, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n4, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n5, a12;\n\t"
            "madc.hi.cc.u32  a13, m, n6, a13;\n\t"
            "madc.hi.cc.u32  a14, m, n7, a14;\n\t"
            "addc.u32  a15, a15, 0;\n\t"

            "mul.lo.u32   m, a7, q;\n\t"
            "mad.lo.cc.u32  a7, m, n0, a7;\n\t"
            "madc.lo.cc.u32  a8, m, n1, a8;\n\t"
            "madc.lo.cc.u32  a9, m, n2, a9;\n\t"
            "madc.lo.cc.u32  a10, m, n3, a10;\n\t"
            "madc.lo.cc.u32  a11, m, n4, a11;\n\t"
            "madc.lo.cc.u32  a12, m, n5, a12;\n\t"
            "madc.lo.cc.u32  a13, m, n6, a13;\n\t"
            "madc.lo.cc.u32  a14, m, n7, a14;\n\t"
            "addc.u32  a15, a15, 0;\n\t"

            "mad.hi.cc.u32  a8, m, n0, a8;\n\t"
            "madc.hi.cc.u32  a9, m, n1, a9;\n\t"
            "madc.hi.cc.u32  a10, m, n2, a10;\n\t"
            "madc.hi.cc.u32  a11, m, n3, a11;\n\t"
            "madc.hi.cc.u32  a12, m, n4, a12;\n\t"
            "madc.hi.cc.u32  a13, m, n5, a13;\n\t"
            "madc.hi.cc.u32  a14, m, n6, a14;\n\t"
            "madc.hi.u32  a15, m, n7, a15;\n\t"
            //pack result back
            "mov.b64         %0, {a8,a9};\n\t"  
            "mov.b64         %1, {a10,a11};\n\t"
            "mov.b64         %2, {a12,a13};\n\t"  
            "mov.b64         %3, {a14,a15};\n\t"
            : "=l"(w.nn[0]), "=l"(w.nn[1]), "=l"(w.nn[2]), "=l"(w.nn[3])
            : "l"(T.nn[0]), "l"(T.nn[1]), "l"(T.nn[2]), "l"(T.nn[3]),
                "l"(T.nn[4]), "l"(T.nn[5]), "l"(T.nn[6]), "l"(T.nn[7]));

    
    if (CMP(w, BASE_FIELD_P) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        w = SUB(w, BASE_FIELD_P);
    }
	
    return w;
}

#define STR_VALUE(arg)  #arg

//This block will be repeated - again and again
#define ASM_REDUCTION_BLOCK \
"mul.lo.u32   m, r0, q;\n\t" \
"mad.lo.cc.u32 r0, m, n0, r0;\n\t" \
"madc.hi.cc.u32 r1, m, n0, r1;\n\t" \
"madc.hi.cc.u32 r2, m, n1, r2;\n\t" \
"madc.hi.cc.u32 r3, m, n2, r3;\n\t" \
"madc.hi.cc.u32 r4, m, n3, r4;\n\t" \
"madc.hi.cc.u32  r5, m, n4, r5;\n\t" \
"madc.hi.cc.u32  r6, m, n5, r6;\n\t" \
"madc.hi.cc.u32  r7, m, n6, r7;\n\t" \
"madc.hi.cc.u32  prefix_low, m, n7, prefix_low;\n\t" \
"addc.u32  prefix_high, 0, 0;\n\t" \
"mad.lo.cc.u32 r0, m, n1, r1;\n\t" \
"madc.lo.cc.u32  r1, m, n2, r2;\n\t" \
"madc.lo.cc.u32  r2, m, n3, r3;\n\t" \
"madc.lo.cc.u32  r3, m, n4, r4;\n\t" \
"madc.lo.cc.u32  r4, m, n5, r5;\n\t" \
"madc.lo.cc.u32  r5, m, n6, r6;\n\t" \
"madc.lo.cc.u32  r6, m, n7, r7;\n\t" \
"addc.cc.u32  r7, prefix_low, 0;\n\t" \
"addc.u32  prefix_low, prefix_high, 0;\n\t"

//This block will also be repeated - but with rising index of a: a1, a2, ..., a7
#define ASM_MUL_BLOCK(idx) \
"mad.lo.cc.u32 r0, a"#idx", b0, r0;\n\t" \
"madc.lo.cc.u32 r1, a"#idx", b1, r1;\n\t" \
"madc.lo.cc.u32 r2, a"#idx", b2, r2;\n\t" \
"madc.lo.cc.u32 r3, a"#idx", b3, r3;\n\t" \
"madc.lo.cc.u32 r4, a"#idx", b4, r4;\n\t" \
"madc.lo.cc.u32 r5, a"#idx", b5, r5;\n\t" \
"madc.lo.cc.u32 r6, a"#idx", b6, r6;\n\t" \
"madc.lo.cc.u32 r7, a"#idx", b7, r7;\n\t" \
"addc.u32 prefix_low, prefix_low, 0;\n\t" \
"mad.hi.cc.u32 r1, a"#idx", b0, r1;\n\t" \
"madc.hi.cc.u32 r2, a"#idx", b1, r2;\n\t" \
"madc.hi.cc.u32 r3, a"#idx", b2, r3;\n\t" \
"madc.hi.cc.u32 r4, a"#idx", b3, r4;\n\t" \
"madc.hi.cc.u32 r5, a"#idx", b4, r5;\n\t" \
"madc.hi.cc.u32 r6, a"#idx", b5, r6;\n\t" \
"madc.hi.cc.u32 r7, a"#idx", b6, r7;\n\t" \
"madc.hi.cc.u32 prefix_low, a"#idx", b7, prefix_low;\n\t" \
"addc.u32 prefix_high, 0, 0;\n\t"

//NB: look carefully on line 11 on page 31 of http://eprints.utar.edu.my/2494/1/CS-2017-1401837-1.pdf
//and find an opportunity for additional speedup
DEVICE_FUNC uint256_g mont_mul_256_asm_CIOS(const uint256_g& u, const uint256_g& v)
{
     uint256_g w;

     asm (  ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
            ".reg .u32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
            ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
            ".reg .u32 n0, n1, n2, n3, n4, n5, n6, n7;\n\t"
            ".reg .u32 m, q, prefix_low, prefix_high;\n\t"
 
            "mov.b64         {a0,a1}, %4;\n\t"
            "mov.b64         {a2,a3}, %5;\n\t"
            "mov.b64         {a4,a5}, %6;\n\t"
            "mov.b64         {a6,a7}, %7;\n\t"
            "mov.b64         {b0,b1}, %8;\n\t"
            "mov.b64         {b2,b3}, %9;\n\t"
            "mov.b64         {b4,b5}, %10;\n\t"
            "mov.b64         {b6,b7}, %11;\n\t"
            "ld.const.u32    n0, [BASE_FIELD_P];\n\t"
            "ld.const.u32    n1, [BASE_FIELD_P + 4];\n\t"
            "ld.const.u32    n2, [BASE_FIELD_P + 8];\n\t"
            "ld.const.u32    n3, [BASE_FIELD_P + 12];\n\t"
            "ld.const.u32    n4, [BASE_FIELD_P + 16];\n\t"
            "ld.const.u32    n5, [BASE_FIELD_P + 20];\n\t"
            "ld.const.u32    n6, [BASE_FIELD_P + 24];\n\t"
            "ld.const.u32    n7, [BASE_FIELD_P + 28];\n\t"
            "ld.const.u32    q, [BASE_FIELD_N];\n\t"

            "mul.lo.u32 r0, a0, b0;\n\t"
            "mul.lo.u32 r1, a0, b1;\n\t"
            "mul.lo.u32 r2, a0, b2;\n\t"
            "mul.lo.u32 r3, a0, b3;\n\t"
            "mul.lo.u32 r4, a0, b4;\n\t"
            "mul.lo.u32 r5, a0, b5;\n\t"
            "mul.lo.u32 r6, a0, b6;\n\t"
            "mul.lo.u32 r7, a0, b7;\n\t"
            "mad.hi.cc.u32 r1, a0, b0, r1;\n\t"
            "madc.hi.cc.u32 r2, a0, b1, r2;\n\t"
            "madc.hi.cc.u32 r3, a0, b2, r3;\n\t"
            "madc.hi.cc.u32 r4, a0, b3, r4;\n\t"
            "madc.hi.cc.u32 r5, a0, b4, r5;\n\t"
            "madc.hi.cc.u32 r6, a0, b5, r6;\n\t"
            "madc.hi.cc.u32 r7, a0, b6, r7;\n\t"
            "madc.hi.cc.u32 prefix_low, a0, b7, 0;\n\t"

            ASM_REDUCTION_BLOCK
            ASM_MUL_BLOCK(1)
            ASM_REDUCTION_BLOCK
            ASM_MUL_BLOCK(2)
            ASM_REDUCTION_BLOCK
            ASM_MUL_BLOCK(3)
            ASM_REDUCTION_BLOCK
            ASM_MUL_BLOCK(4)
            ASM_REDUCTION_BLOCK
            ASM_MUL_BLOCK(5)
            ASM_REDUCTION_BLOCK
            ASM_MUL_BLOCK(6)
            ASM_REDUCTION_BLOCK
            ASM_MUL_BLOCK(7)
            ASM_REDUCTION_BLOCK
    
            //pack result back
            "mov.b64         %0, {r0,r1};\n\t"  
            "mov.b64         %1, {r2,r3};\n\t"
            "mov.b64         %2, {r4,r5};\n\t"  
            "mov.b64         %3, {r6,r7};\n\t"
            : "=l"(w.nn[0]), "=l"(w.nn[1]), "=l"(w.nn[2]), "=l"(w.nn[3])
            : "l"(u.nn[0]), "l"(u.nn[1]), "l"(u.nn[2]), "l"(u.nn[3]),
                "l"(v.nn[0]), "l"(v.nn[1]), "l"(v.nn[2]), "l"(v.nn[3]));
                                   
    if (CMP(w, BASE_FIELD_P) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        w = SUB(w, BASE_FIELD_P);
    }
	
    return w;
}