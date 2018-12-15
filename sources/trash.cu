__device__ inline uint512_g mul_uint256_to_512_asm(const uint256_g& u, const uint256_g& v)
{
    my_uint128_t res;
    asm ("{\n\t"
         "mul.lo.u32      %0, %8, %16;    \n\t"
         "mul.hi.u32      %1, %8, %16;    \n\t"
         "mad.lo.cc.u32   %1, %9, %16, %1;\n\t"
         "madc.hi.u32     %2, %9, %16,  0;\n\t"
         "mad.lo.cc.u32   %1, %8, %17, %1;\n\t"
         "madc.hi.cc.u32  %2, %8, %17, %2;\n\t"
         "madc.hi.u32     %3, %8,%17,  0;\n\t"
         "mad.lo.cc.u32   %2, %8,%10, %2;\n\t"
         "madc.hi.u32     %3, %5, %9, %3;\n\t"
         "mad.lo.cc.u32   %2, %5, %9, %2;\n\t"
         "madc.hi.u32     %3, %6, %8, %3;\n\t"
         "mad.lo.cc.u32   %2, %6, %8, %2;\n\t"
         "madc.lo.u32     %3, %4,%11, %3;\n\t"
         "mad.lo.u32      %3, %5,%10, %3;\n\t"
         "mad.lo.u32      %3, %6, %9, %3;\n\t"
         "mad.lo.u32      %3, %7, %8, %3;\n\t"
         "}"
         : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
         : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w),
           "r"(b.x), "r"(b.y), "r"(b.z), "r"(b.w));
    return res;
}





//let u = u1 + u2 * 2^n
//let v = v1 + v2 * 2^n
//result = (u1 * v1) + 2^n * ((u1 + u2)(v1 + v2) - (u1 * v1)(u2 * v2))  + 2^(2n) u2 * v2;
//hence we require more space and addition operations bu only 3 multiplications (instead of four)


__device__ inline uint512_g mul_uint256_to_512_Karatsuba(const uint256_g& u, const uint256_g& v)
{
    uint256 x = ;
		
    #pragma unroll
	for (uint32_t j = 0; j < 8; j++)
	{
        uint32_t i = 0;
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

        w.n[N + j] = k;
    }

    return w;	
}


def is_in_range(x, shift, array_len):
    flag = (x >= shift and x < array_len + shift)
    return flag

def gen_sublists(idx, array_len):
    sublists = []
    if (idx > 0):
        for j in xrange(min(idx, array_len)):
            a = idx - 1 - j + 2 * array_len
            b = j + 3 * array_len
            if is_in_range(a, 2 * array_len, array_len) and is_in_range(b, 3 * array_len, array_len):
                sublists.append((a, b, True))
    for j in xrange(min(idx, array_len) + 1):
        a = idx - j + 2 * array_len
        b = j + 3 * array_len
        if is_in_range(a, 2 * array_len, array_len) and is_in_range(b, 3 * array_len, array_len):
            sublists.append((a, b, False))
    return sublists

from collections import namedtuple

#c = a*b + d
AsmInsn = namedtuple("AsmInsn", "gen_carry use_carry with_addition is_high a b c d")

def gen_table(array_len):
    table = {}
    table_len = 0
    for i in xrange(array_len * 2):
        arr = gen_sublists(i, array_len)
        table_len += len(arr)
        table[i] = arr
    return table, table_len


def gen_asm(array_len):
    
    carry_arr = [False] * (2 * array_len)
    AsmListing = []
    
    table, table_len = gen_table(array_len)
    print table_len
    lowest_index = 0
    cur_index = 0
    use_carry = False
    while(table_len): 
        if carry_arr[cur_index]:
            with_addition = True
            gen_carry = True
            
            (a, b, is_high) = table[cur_index][0]
            table[cur_index].pop(0)
            table_len = table_len - 1
            
            insn = AsmInsn(gen_carry, use_carry, with_addition, is_high, a, b, cur_index, cur_index)
            AsmListing.append(insn)
            
            use_carry = True
            cur_index = cur_index + 1
            
        else:
            with_addition = use_carry
            gen_carry = False
            
            (a, b, is_high) = table[cur_index][0]
            table[cur_index].pop(0)
            table_len = table_len - 1
            
            insn = AsmInsn(gen_carry, use_carry, with_addition, is_high, a, b, cur_index, -1)
            AsmListing.append(insn)
            
            carry_arr[cur_index] = True
            use_carry = False
            
            if table_len == 0:
                break
            #try to find next suitable index
            while not table[lowest_index]:
                lowest_index = lowest_index + 1
            cur_index = lowest_index
        
    return AsmListing


def generate_printable_asm(AsmListing):
    printed_asm = ""
    for elem in AsmListing:
        high_low = "hi." if elem.is_high else "lo."
        if not elem.with_addition:          
            printed_asm += "mul." + high_low + "u32" + '   %{:d}, %{:d}, %{:d};\n'.format(elem.c, elem.a, elem.b)
        else:
            printed_asm += "mad"
            if (elem.use_carry):
                printed_asm += "c"
            printed_asm += "." + high_low
            if (elem.gen_carry):
                printed_asm += "cc."
            printed_asm += "u32" + '   %{:d}, %{:d}, %{:d}, '.format(elem.c, elem.a, elem.b)
            ending = "0;\n" if elem.d == -1 else '%{:d};\n'.format(elem.d)
            printed_asm += ending
     
    return printed_asm

    

AsmListing = gen_asm(3)
print generate_printable_asm(AsmListing)


//-------------------------------------------


#include "cuda_structs.h"
//helper functions for naive multiplication

DEVICE_FUNC inline uint32_t device_long_mul(uint32_t x, uint32_t y, uint32_t* high_ptr)
	{
		uint32_t high = __umulhi(x, y);
		*high_ptr = high;
		return x * y;
	}

DEVICE_FUNC static uint32_t device_fused_add(uint32_t x, uint32_t y, uint32_t* high_ptr)
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
	for (uint32_t j = 0; j < 4; j++)
	{
        uint32_t k = 0;

        #pragma unroll
        for (uint32_t i = 0; i < 4; i++)
        {
            uint32_t high_word = 0;
            uint32_t low_word = 0;
            low_word = device_long_mul(u.n[i], v.n[j], &high_word);
            low_word = device_fused_add(low_word, w.n[i + j], &high_word);
            low_word = device_fused_add(low_word, k, &high_word);
            k = high_word;
            w.n[i + j] = low_word;
        }

        w.n[4 + j] = k;
    }

    return w;	
}

//this code was produced by my generator
DEVICE_FUNC inline uint256_g mul_uint128_to_256_asm_ver1(const uint128_g& lhs, const uint128_g& rhs)
{
     uint256_g w;
      asm (     "mul.lo.u32   %0, %8, %12;\n\t"
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
         : "=l"(res.a), "=l"(res.b), "=l"(res.c), "=l"(res.d)
         : "l"(a.low), "l"(a.high), "l"(b.low), "l"(b.high));

    return res;
}

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
         : "=l"(res.a), "=l"(res.b), "=l"(res.c), "=l"(res.d)
         : "l"(a.low), "l"(a.high), "l"(b.low), "l"(b.high));

    return res;
}
#endif

#define FASTEST_128_to_256_mul(a, b) mul_uint128_to_256_asm_ver1(a, b)






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


//benchmarking code
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

uint256_g get_random_elem()
{
    uint256_g res;
    for (uint32_t i =0; i < 8; i++)
        res.n[i] = rand();
    return res;
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

__global__ void add_kernel_native(uint256_g* a_arr, uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = sub_uint256_naive(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void add_kernel_asm(uint256_g* a_arr, uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = sub_uint256_asm(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

//----------------------------------------------------------------------------------------------------------------------

__global__ void mul_kernel_naive(uint256_g* __restrict__  a_arr, uint256_g* __restrict__  b_arr, uint512_g* __restrict__  c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = mul_uint256_to_512_naive(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void mul_kernel_asm(uint256_g* __restrict__ a_arr, uint256_g* __restrict__  b_arr,  uint512_g* __restrict__ c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = mul_uint256_to_512_asm(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}




void gpu_benchmark(size_t bench_len)
{
    using Atype = uint256_g;
    using Btype = uint256_g;
    using Ctype = uint512_g;
    
    Atype* A_host_arr = nullptr;
    Btype* B_host_arr = nullptr;
    Ctype* C_host_arr = nullptr;

    Atype* A_dev_arr = nullptr;
    Btype* B_dev_arr = nullptr;
    Ctype* C_dev_arr = nullptr;

    std::chrono::high_resolution_clock::time_point start, end;
    std::int64_t duration1, duration2, duration3, duration4;

    cudaError_t cudaStatus;

    //fill in A array
    A_host_arr = (Atype*)malloc(bench_len * sizeof(Atype));

    for (size_t i = 0; i < bench_len; i++)
    {
        A_host_arr[i] = get_random_elem();

    }

    //fill in B array
    B_host_arr = (Btype*)malloc(bench_len * sizeof(Btype));
    for (size_t i = 0; i < bench_len; i++)
    {
        B_host_arr[i] = get_random_elem();
    }

    //allocate C array
    C_host_arr = (Ctype*)malloc(bench_len * sizeof(Ctype));

    cudaStatus = cudaMalloc(&A_dev_arr, bench_len * sizeof(Atype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (A_dev_arr) failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&B_dev_arr, bench_len * sizeof(Btype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (B_dev_arr) failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&C_dev_arr, bench_len * sizeof(Ctype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (C_dev_arr) failed!");
        goto Error;
    }

    

    cudaStatus = cudaMemcpy(A_dev_arr, A_host_arr, bench_len * sizeof(Atype), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (A_arrs) failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(B_dev_arr, B_host_arr, bench_len * sizeof(Btype), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (B_arrs) failed!");
        goto Error;
    }

    //run_kernel and measure!
    //---------------------------------------------------------------------------------------------------------------------------------
    for (unsigned i = 0; i < 2; i++)
    {
        start = std::chrono::high_resolution_clock::now();
        switch (i)
        {
        case 0:
           mul_kernel_naive<<<1024, 64>>>(A_dev_arr, B_dev_arr, C_dev_arr, bench_len);
            
            break;
        case 1:
            mul_kernel_asm<<<1024, 64>>>(A_dev_arr, B_dev_arr, C_dev_arr, bench_len);
            break;
        }
        
         // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();

    
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        end = std::chrono::high_resolution_clock::now();
        switch (i)
        {
        case 0:
            duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
            break;
        case 1:
            duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
            break;
        }
    }

    std::cout << "ns total GPU naive: " << duration1  << "ns." << std::endl;
    std::cout << "ns total GPU asm: " << duration2  << "ns." << std::endl;
  
Error:
    cudaFree(A_dev_arr);
    cudaFree(B_dev_arr);
    cudaFree(C_dev_arr);

    free(A_host_arr);
    free(B_host_arr);
    free(C_host_arr);
}





int main(int argc, char* argv [])
{
   gpu_benchmark(2000000);
}


        
        
        
        
        
        
        
        
        
        
        
        
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

"mul.lo.u32   %0, %8, %12;\n\t"
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
"addc.cc.u32   %5, %5, 0;\n\t"
"addc.cc.u32   %6, %6, 0;\n\t"
"addc.cc.u32   %7, %7, 0;\n\t"