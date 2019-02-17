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


#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <semaphore.h> 
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

#define SEM_NAME "/qwerty"
#define SEM_SET 0
#define BUF_LEN 2048
#define TEST_LEN 10

#define SAGE_DIR
#define SAGE_EXECUTABLE




int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: correctness_test port");
    }

    long port = strtol(argv[1], NULL, 10);
    
    sem_t* sem = sem_open(SEM_NAME, O_CREAT | O_EXCL, 0700, SEM_SET);
    if (sem == SEM_FAILED)
    {
        perror("semaphore");
        exit(1);
    }

    if (fork())
    {
        //create socket server
        int sock, listener;
        struct sockaddr_in addr;
        char buf[BUF_LEN];
        int bytes_read;

        listener = socket(AF_INET, SOCK_STREAM, 0);
        if(listener < 0)
        {
            perror("socket");
            exit(2);
        }
    
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if(bind(listener, (struct sockaddr *)&addr, sizeof(addr)) < 0)
        {
            perror("bind");
            exit(3);
        }

        listen(listener, 1);

        //everything is ready: release mutex
        sem_post(sem);
    
        sock = accept(listener, NULL, NULL);
        if(sock < 0)
        {
            perror("accept");
            exit(4);
        }

        //addition test!
        bytes_read = recv(sock, buf, BUF_LEN, 0);
        for (size_t i = 0; i < TEST_LEN; i++)
        {

        }
                if(bytes_read <= 0) break;
                send(sock, buf, bytes_read, 0);
            }
    
            close(sock);
        }
    } 
    else
    {
        sem_wait(sem);
        //here we execute sage
        execl("/bin/sh", "sh", "-c", command, (char *) 0);
    }
}

#ifndef ELL_POINT_CUH
#define ELL_POINT_CUH

#include "mont_mul.cuh"

//Again we initialize global variables with BN_256 specific values
// A = 0, B = 3, G = [1, 2, 1]

DEVICE_VAR CONST_MEMORY uint256_g A_g = {
    0, 0, 0, 0, 0, 0, 0, 0
};


DEVICE_VAR CONST_MEMORY uint256_g B_g = R3_g;

struct ec_point
{
    uint256_g x;
    uint256_g y;
    uint256_g z;
};

DEVICE_VAR CONST_MEMORY ec_point G = {R_g, R2_g, R_g}




//Implementation of these routines doesn't depend on whether we consider prokective or jacobian coordinates
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC inline bool is_infinity(const ec_point& point)
{	
	return is_zero(point.z);
}

DEVICE_FUNC inline ec_point point_at_infty()
{
    ec_point pt;
	
	//TD: may be we should use asm and xor here)
	#pragma unroll
    for (int32_t i = 0 ; i < N; i++)
    {
        pt.x.n[i] = 0;
    }
    pt.y.n[0] = 1;
	#pragma unroll
    for (int32_t  i= 1 ; i < N; i++)
    {
        pt.y.n[i] = 0;
    }
	#pragma unroll
    for (int32_t i = 0 ; i < N; i++)
    {
        pt.z.n[i] = 0;
    }

	return pt;
}

DEVICE_FUNC inline uint256_g FIELD_ADD(const uint256_g& a, const uint256_g& b)
{
	uint256_g w = ADD(a, b);
	if (CMP(w, modulus_g) >= 0)
		return SUB(w, modulus_g);
	return w;
}

DEVICE_FUNC inline uint256_g FIELD_SUB(const uint256_g& a, const uint256_g& b)
{
	if (CMP(a, b) > 0)
		return SUB(a, b);
	else
	{
		uint256_g t = ADD(a, modulus_g);
		return SUB(t, b);
	}
}

DEVICE_FUNC inline uint256_g FIELD_INV(const uint256_g& elem)
{
	if (!is_zero(elem))
		return SUB(modulus_g, elem);
	else
		return elem;
}

DEVICE_FUNC inline ec_point INV(const ec_point& pt)
{
	ec_point res{pt.x, FIELD_INV(pt.y), pt.z};
}

//Arithmetic in projective coordinates (Jacobian coordinates should be faster and we are going to check it!)
//TODO: we may also use BN specific optimizations (for example use, that a = 0)
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC inline ec_point double_point_proj(const ec_point& pt)
{
	if (is_zero(pt.y))
		return point_at_infty();
	else
	{
		uint256_g temp, temp2;
		uint256_g W, S, B, H, S2;
		ec_point res;

#ifdef BN256_SPECIFIC_OPTIMIZATION
		temp = MONT_SQUARE(pt.x);
		W = MONT_MUL(temp, R3_g);
#else
		temp = MONT_SQUARE(pt.x);
		temp = MONT_MUL(temp, R3_g);
		temp2 = MONT_SQUARE(pt.z);
		temp2 = MONT_MUL(temp2, A_g);
		W = FIELD_ADD(temp, temp2);
#endif
		S = MONT_MUL(pt.y, pt.z);
		temp = MONT_MUL(pt.x, pt.y);
		B = MONT_MUL(temp, S);

		temp = MONT_SQUARE(W);
		temp2 = MONT_MUL(R8_g, B);
		H = FIELD_SUB(temp, temp2);

		temp = MONT_MUL(R2_g, H);
		res.x = MONT_MUL(temp, S);
		
		//NB: here result is also equal to one of the operands and hence may be reused!!!
		//NB: this is in fact another possibility for optimization!
		S2 = MONT_SQUARE(S);
		temp = MONT_MUL(R4_g, B);
		temp = FIELD_SUB(temp, H);
		temp = MONT_MUL(W, temp);
		
		temp2 = MONT_SQUARE(pt.y);
		temp2 = MONT_MUL(R8_g, temp2);
		temp2 = MONT_MUL(temp2, S2);
		res.y = FIELD_SUB(temp, temp2);

		temp = MONT_MUL(R8_g, S);
		res.z = MONT_MUL(temp, S2);

		return res;
	}
}

//for debug purposes only: check if point is indeed on curve
DEVICE_FUNC inline bool check_if_on_curve_proj(const ec_point& pt)
{
	//y^{2} * z = x^{3} + A *x * z^{2} + B * z^{3}
	uint256_g temp1, temp2, z2; 
	z2 = MONT_SQUARE(pt.z);
	temp1 = MONT_SQUARE(pt.x);
	temp1 = MONT_MUL(temp1, pt.x);
	temp2 = MONT_MUL(A_g, pt.x);
	temp2 = MONT_MUL(temp2, z2);
	temp1 = FIELD_ADD(temp1, temp2);
	temp2 = MONT_MUL(B_g, pt.z);
	temp2 = MONT_MUL(temp2, z2);
	temp1 = FIELD_ADD(temp1, temp2);
	temp2 = MONT_SQUARE(pt.y);
	temp2 = MONT_MUL(temp2, pt.z);

	return EQUAL(temp1, temp2);
}

DEVICE_FUNC inline bool equal_proj(const ec_point& pt1, const ec_point& pt2)
{
	//check all of the following equations:
	//X_1 * Y_2 = Y_1 * X_2;
	//X_1 * Z_2 =  X_2 * Y_1;
	//Y_1 * Z_2 = Z_1 * Y_2;
	
	uint256_g temp1, temp2;

	temp1 = MONT_MUL(pt1.x, pt2.y);
	temp2 = MONT_MUL(pt1.y, pt2.x);
	bool first_check = EQUAL(temp1, temp2);

	temp1 = MONT_MUL(pt1.y, pt2.z);
	temp2 = MONT_MUL(pt1.z, pt2.y);
	bool second_check = EQUAL(temp1, temp2);
	
	temp1 = MONT_MUL(pt1.x, pt2.z);
	temp2 = MONT_MUL(pt1.z, pt2.x);
	bool third_check = EQUAL(temp1, temp2);
			
	return (first_check && second_check && third_check);
}

DEVICE_FUNC inline ec_point add_proj(const ec_point& left, const ec_point& right)
{
	if (is_infinity(left))
		return right;
	if (is_infinity(right))
		return left;

	uint256_g U1, U2, V1, V2;
	U1 = MONT_MUL(left.z, right.y);
	U2 = MONT_MUL(left.y, right.z);
	V1 = MONT_MUL(left.z, right.x);
	V2 = MONT_MUL(left.x, right.z);

	ec_point res;

	if (EQUAL(V1, V2))
	{
		if (!EQUAL(U1, U2))
			return point_at_infty();
		else
			return  double_point_proj(left);
	}

	uint256_g U = FIELD_SUB(U1, U2);
	uint256_g V = FIELD_SUB(V1, V2);
	uint256_g W = MONT_MUL(left.z, right.z);
	uint256_g Vsq = MONT_SQUARE(V);
	uint256_g Vcube = MONT_MUL(Vsq, V);

	uint256_g temp1, temp2;
	temp1 = MONT_SQUARE(U);
	temp1 = MONT_MUL(temp1, W);
	temp1 = FIELD_SUB(temp1, Vcube);
	temp2 = MONT_MUL(R2_g, Vsq);
	temp2 = MONT_MUL(Vsq, V2);
	uint256_g A = FIELD_SUB(temp1, temp2);
	res.x = MONT_MUL(V, A);

	temp1 = MONT_MUL(Vsq, V2);
	temp1 = FIELD_SUB(temp1, A);
	temp1 = MONT_MUL(U, temp1);
	temp2 = MONT_MUL(Vcube, U2);
	res.y = FIELD_SUB(temp1, temp2);

	res.z = MONT_MUL(Vcube, W);
	return res;
}

DEVICE_FUNC inline ec_point sub_proj(const ec_point& left, const ec_point& right)
{
	return add_proj(left, INV(right));
}

//Arithmetic in Jacobian coordinates (Jacobian coordinates should be faster and we are going to check it!)
//TODO: we may also use BN specific optimizations (for example use, that a = 0)
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC inline ec_point double_point_jac(const ec_point& pt)
{
	if (is_zero(pt.y))
		return point_at_infty();
	else
	{
		uint256_g temp1, temp2;
		temp1 = MONT_MUL(R4_g, pt.x);
		uint256_g Ysq = MONT_SQUARE(pt.y);
		uint256_g S = MONT_MUL(temp1, Ysq);

		temp1 = MONT_SQUARE(pt.x);
		temp1 = MONT_MUL(R3_g, temp1);
		temp2 = MONT_SQUARE(pt.z);
		temp2 = MONT_SQUARE(temp2);
		uint256_g M = FIELD_ADD(temp1, temp2);

		temp1 = MONT_SQUARE(M);
		temp2 = MONT_MUL(R2_g, S);
		uint256_g res_x = FIELD_SUB(temp1, temp2);
		
		temp1 = FIELD_SUB(S, res_x);
		temp1 = MONT_MUL(M, temp1);
		temp2 = MONT_SQUARE(Ysq);
		temp2 = MONT_MUL(R8_g, temp2);
		uint256_g res_y = FIELD_SUB(temp1, temp2);

		temp1 = MONT_MUL(R2_g, pt.y);
		uint256_g res_z = MONT_MUL(temp1, pt.z);

		return ec_point{res_x, res_y, res_z};
	}
}

DEVICE_FUNC inline bool check_if_on_curve_jac(const ec_point& pt)
{
	//y^4 = x^3 + a  x z^4 +b z^6
	uint256_g temp1 = MONT_SQUARE(pt.y);
	uint256_g lefthandside = MONT_SQUARE(temp1);

	uint256_g Zsq = MONT_SQUARE(pt.z);
	uint256_g Z4 = MONT_SQUARE(Zsq);

	temp1 = MONT_SQUARE(pt.x);
	uint256_g righthandside = MONT_MUL(temp1, pt.x);
	temp1 = MONT_MUL(A_g, pt.x);
	temp1 = MONT_MUL(temp1, Z4);
	righthandside = FIELD_ADD(righthandside, temp1);
	temp1 = MONT_MUL(B_g, Zsq);
	temp1 = MONT_MUL(temp1, Z4);
	righthandside = FIELD_ADD(righthandside, temp1);
}

DEVICE_FUNC inline bool equal_jac(const ec_point& pt1, const ec_point& pt2)
{
	if (is_infinity(pt1) ^ is_infinity(pt2))
		return false;
	if (is_infinity(pt1) & is_infinity(pt2))
		return true;

	//now both points are not points at infinity.

	uint256_g Z1sq = MONT_SQUARE(pt1.z);
	uint256_g Z2sq = MONT_SQUARE(pt2.z);

	uint256_g temp1 = MONT_MUL(pt1.x, Z2sq);
	uint256_g temp2 = MONT_MUL(pt2.x, Z1sq);
	bool first_check = EQUAL(temp1, temp2);

	temp1 = MONT_MUL(pt1.y, Z2sq);
	temp1 = MONT_MUL(temp1, pt2.z);
	temp2 = MONT_MUL(pt2.y, Z1sq);
	temp2 = MONT_MUL(temp2, pt2.z);
	bool second_check = EQUAL(temp1, temp2);

	return (first_check && second_check);
}

DEVICE_FUNC inline ec_point add_jac(const ec_point& left, const ec_point& right)
{
	if (is_infinity(left))
		return right;
	if (is_infinity(right))
		return left;

	uint256_g U1, U2;

	uint256_g Z2sq = MONT_SQUARE(right.z);
	U1 = MONT_MUL(left.x, Z2sq);

	uint256_g Z1sq = MONT_SQUARE(left.z);
	U2 = MONT_MUL(right.x, Z1sq);

	uint256_g S1 = MONT_MUL(left.y, Z2sq);
	S1 = MONT_MUL(S1, right.z);

	uint256_g S2 = MONT_MUL(right.y, Z1sq);
	S2 = MONT_MUL(S2, left.z);

	if (EQUAL(U1, U2))
	{
		if (!EQUAL(S1, S2))
			return point_at_infty();
		else
			return  double_point_proj(left);
	}

	uint256_g H = FIELD_SUB(U2, U1);
	uint256_g R = FIELD_SUB(S2, S1);
	uint256_g Hsq = MONT_SQUARE(H);
	uint256_g Hcube = MONT_MUL(Hsq, H);
	uint256_g T = MONT_MUL(U1, Hsq);

	uint256_g res_x = MONT_SQUARE(R);
	res_x = FIELD_SUB(res_x, Hcube);
	uint256_g temp = MONT_MUL(R2_g, T);
	res_x = FIELD_SUB(res_x, temp);

	uint256_g res_y = FIELD_SUB(T, res_x);
	res_y = MONT_MUL(R, res_y);
	temp = MONT_MUL(S1, Hcube);

	uint256_g res_z = MONT_MUL(H, left.z);
	res_z = MONT_MUL(res_z, right.z);

	return ec_point{res_x, res_y, res_z};
}

DEVICE_FUNC inline ec_point sub_jac(const ec_point& left, const ec_point& right)
{
	return add_jac(left, INV(right));
}

#ifdef PROJ_COORDINATES
#define EC_ADD(x, y) add_proj(x, y)
#define EC_SUB(x, y) sub_proj(x, y)
#define EC_DOUBLE(x) double_point_proj(x)
#define IS_ON_CURVE(x) check_if_on_curve_proj(x)
#else
#define EC_ADD(x, y) add_jac(x, y)
#define EC_SUB(x, y) sub_jac(x, y)
#define EC_DOUBLE(x) double_point_jac(x)
#define IS_ON_CURVE(x) check_if_on_curve_jac(x)
#endif

#endif

p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

base_field = GF(p)
curve = EllipticCurve(GF(p), [0, 3]);
G = curve(1, 2, 1)

R = field(2 ^ 256)

/*#ifdef PROJ_COORDINATES
#define EC_ADD(x, y) add_proj(x, y)
#define EC_SUB(x, y) sub_proj(x, y)
#define EC_DOUBLE(x) double_point_proj(x)
#define IS_ON_CURVE(x) check_if_on_curve_proj(x)
#else
#define EC_ADD(x, y) add_jac(x, y)
#define EC_SUB(x, y) sub_jac(x, y)
#define EC_DOUBLE(x) double_point_jac(x)
#define IS_ON_CURVE(x) check_if_on_curve_jac(x)
#endif*/


#define GENERAL_TEST(func_name) \
__global__ void func_name##_kernel(uint256_g* a_arr, uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(uint256_g* a_arr, uint256_g* b_arr, uint256_g* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}


#define MUL_TEST(func_name) \
__global__ void func_name##_kernel(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}

#define SQUARE_TEST(func_name) \
__global__ void func_name##_kernel(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(uint256_g* a_arr, uint256_g* b_arr, uint512_g* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}

#define ECC_TEST(func_name) \
__global__ void func_name##_kernel(ec_point* a_arr, ec_point* b_arr, ec_point* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid], b_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(ecc_point* a_arr, ecc_point* b_arr, ecc_point* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}

#define ECC_DOUBLE_TEST(func_name) \
__global__ void func_name##_kernel(ec_point* a_arr, ec_point* b_arr, ec_point* c_arr, size_t arr_len)\
{\
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;\
	while (tid < arr_len)\
	{\
		c_arr[tid] = func_name(a_arr[tid]);\
		tid += blockDim.x * gridDim.x;\
	}\
}\
\
void func_name##_driver(ecc_point* a_arr, ecc_point* b_arr, ecc_point* c_arr, size_t arr_len)\
{\
	func_name##_kernel<<<4096, 248>>>(a_arr, b_arr, c_arr, arr_len);\
}


using mul_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint512_g>;
using general_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint256_g>;
using ell_point_func_vec_t = kernel_func_vec_t<ec_point, ec_point, ec_point>;

GENERAL_TEST(add_uint256_naive)
GENERAL_TEST(add_uint256_asm)

general_func_vec_t addition_bench = {
    {"naive approach", add_uint256_naive_driver},
	{"asm", add_uint256_asm_driver}
};

GENERAL_TEST(sub_uint256_naive)
GENERAL_TEST(sub_uint256_asm)

general_func_vec_t substraction_bench = {
    {"naive approach", sub_uint256_naive_driver},
	{"asm", sub_uint256_asm_driver}
};

MUL_TEST(mul_uint256_to_512_asm)
MUL_TEST(mul_uint256_to_512_naive)
MUL_TEST(mul_uint256_to_512_asm_with_allocation)
MUL_TEST(mul_uint256_to_512_asm_longregs)
MUL_TEST(mul_uint256_to_512_Karatsuba)

mul_func_vec_t mul_bench = {
    {"naive approach", mul_uint256_to_512_naive_driver},
	{"asm", mul_uint256_to_512_asm_driver},
	{"asm with register alloc", mul_uint256_to_512_asm_with_allocation_driver},
	{"asm with longregs", mul_uint256_to_512_asm_longregs_driver},
    {"Karatsuba", mul_uint256_to_512_Karatsuba_driver}
};

SQUARE_TEST(square_uint256_to_512_naive)
SQUARE_TEST(square_uint256_to_512_asm)




#include "benchmark.cuh"
#include "func_lists.cuh"

#include <stdio.h>
#include <time.h>


GENERAL_TEST_1_ARG_1_TYPE(ECC_DOUBLE_PROJ, ec_point);


size_t bench_len = 0x3;

int main(int argc, char* argv[])
{
	
	//long ltime = time (NULL);
    //unsigned int stime = (unsigned int) ltime/2;
    //srand(stime);
    
    //gpu_benchmark(mul_bench, bench_len);
	
	//gpu_benchmark(square_bench, bench_len);
	
	//gpu_benchmark(mont_mul_bench, bench_len);
    return 0;
}

mul_func_vec_t square_bench = {
    {"naive approach", square_uint256_to_512_naive_driver},
	{"asm", square_uint256_to_512_asm_driver},
};


GENERAL_TEST(mont_mul_256_naive_SOS)
GENERAL_TEST(mont_mul_256_naive_CIOS)
GENERAL_TEST(mont_mul_256_asm_SOS)
GENERAL_TEST(mont_mul_256_asm_CIOS)

general_func_vec_t mont_mul_bench = {
    {"naive SOS", mont_mul_256_naive_SOS_driver},
	{"asm SOS", mont_mul_256_asm_SOS_driver},
	{"naive CIOS", mont_mul_256_naive_CIOS_driver},
	{"asm CIOS", mont_mul_256_asm_CIOS_driver}
};




//--------------------------------------------------------------------------------------------------------------------------------------------



#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <vector>



std::ostream& operator<<(std::ostream& os, const uint256_g num)
{
    os << "0x";
    for (int i = 7; i >= 0; i--)
    {
        os << std::setfill('0') << std::hex << std::setw(8) << num.n[i];
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const uint512_g num)
{
    os << "0x";
    for (int i = 15; i >= 0; i--)
    {
        os << std::setfill('0') << std::hex << std::setw(8) << num.n[i];
    }
    return os;
}



//----------------------------------------------------------------------------------------------------------------------------------------------


p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

base_field = GF(p)
curve = EllipticCurve(GF(p), [0, 3]);
G = curve(1, 2, 1)

R = base_field(2 ^ 256)

def mont_mul(x, y):
    return int(base_field(x) * base_field(y) / R)


def to_mont_form(x):
    return (base_field(x) * R)

def from_mont_form(x):
    return (base_field(x) / R)



# A = curve(ax1, ay1, az1)

# B = curve(bx1, by1, bz1)

# C = curve(cx1, cy1, cz1)

# # D = curve(ex1, ey1, ez1)

u = from_mont_form(dx1)
v = from_mont_form(dy1)
w = from_mont_form(dz1)

C_jac = curve(u / (w^2), v / (w^3))

u = from_mont_form(fx1)
v = from_mont_form(fy1)
w = from_mont_form(fz1)

D_jac = curve(u / (w^2), v / (w^3))

# print A + B == C_jac
# print A - B == D_jac
# print A - B == D

#B = curve(a / (c^2), b / (c^3))


#B = curve(mont_mul(x7, z7), y7, mont_mul(mont_mul(z7, z7), z7))




import math

p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
field = GF(p)

D = -3

#(Represent 4p as x2 + |D|y2 (modified Cornacchia–Smith)) 
#Given a prime p > 2 and −4p<D< 0 with D ≡ 0, 1 (mod 4), this algorithm
#either reports that no solution exists, or returns a solution (x, y).

def Cornacchia_Smith_alg(p, d):
    #[Test for solvability]
    if kronecker(d, p) < 1:
        return False
    
    #[Initial square root]
    x_0 = int(field(d).sqrt())
    if (x_0 - d) % 2:
        x_0 = p - x_0
        
    #[Initialize Euclid chain]
    (a, b) = (2 * p, x_0);
    c = math.floor(2 * math.sqrt(p))
    
    #[Euclid chain]
    while (b > c): 
        (a, b) = (b, a % b)

    #[Final report]
    t = 4 * p - b ^ 2;
    if (t % d != 0):
        return False
    y = t / abs(D)
    if sqrt(y) not in QQ:
        return False

    return (b, sqrt(y))

#find if n is represented as m * r, where r is a large prime number

def Is_large(n):
    factors = n.factor()
    for elem in factors:
        if len(bin(elem[0])) >= 242:
            return True
    return False




while (D > -100):
    if (Cornacchia_Smith_alg(p, D)):
        
        (u, v) = Cornacchia_Smith_alg(p, D)
        n1 = p + 1 + u
        n2 = p + 1 - u
        if Is_large(n1) or Is_large(n2):
            print D
    D = D - 1
print "end" 


mul.wide.u16    r0, a0, b0;\n\t
mul.wide.u16    r1, a0, b2;\n\t
mul.wide.u16    r2, a0, b4;\n\t
mul.wide.u16    r3, a0, b6;\n\t
mul.wide.u16    r4, a1, b7;\n\t
mul.wide.u16    r5, a3, b7;\n\t
mul.wide.u16    r6, a5, b7;\n\t
mul.wide.u16    r7, a7, b7;\n\t
mul.wide.u16    t3, a1, b5;\n\t
mul.wide.u16    t4, a2, b6;\n\t
add.cc.u32    r3, r3, t3;\n\t
addc.cc.u32    r4, r4, t4;\n\t
addc.u32    r5, r5, 0;\n\t
mul.wide.u16    t3, a2, b4;\n\t
mul.wide.u16    t4, a3, b5;\n\t
add.cc.u32    r3, r3, t3;\n\t
addc.cc.u32    r4, r4, t4;\n\t
addc.u32    r5, r5, 0;\n\t
mul.wide.u16    t2, a1, b3;\n\t
mul.wide.u16    t3, a3, b3;\n\t
mul.wide.u16    t4, a4, b4;\n\t
mul.wide.u16    t5, a4, b6;\n\t
add.cc.u32    r2, r2, t2;\n\t
addc.cc.u32    r3, r3, t3;\n\t
addc.cc.u32    r4, r4, t4;\n\t
addc.cc.u32    r5, r5, t5;\n\t
addc.u32    r6, r6, 0;\n\t
mul.wide.u16    t2, a2, b2;\n\t
mul.wide.u16    t3, a4, b2;\n\t
mul.wide.u16    t4, a5, b3;\n\t
mul.wide.u16    t5, a5, b5;\n\t
add.cc.u32    r2, r2, t2;\n\t
addc.cc.u32    r3, r3, t3;\n\t
addc.cc.u32    r4, r4, t4;\n\t
addc.cc.u32    r5, r5, t5;\n\t
addc.u32    r6, r6, 0;\n\t
mul.wide.u16    t1, a1, b1;\n\t
mul.wide.u16    t2, a3, b1;\n\t
mul.wide.u16    t3, a5, b1;\n\t
mul.wide.u16    t4, a6, b2;\n\t
mul.wide.u16    t5, a6, b4;\n\t
mul.wide.u16    t6, a6, b6;\n\t
add.cc.u32    r1, r1, t1;\n\t
addc.cc.u32    r2, r2, t2;\n\t
addc.cc.u32    r3, r3, t3;\n\t
addc.cc.u32    r4, r4, t4;\n\t
addc.cc.u32    r5, r5, t5;\n\t
addc.cc.u32    r6, r6, t6;\n\t
addc.u32    r7, r7, 0;\n\t
mul.wide.u16    t1, a2, b0;\n\t
mul.wide.u16    t2, a4, b0;\n\t
mul.wide.u16    t3, a6, b0;\n\t
mul.wide.u16    t4, a7, b1;\n\t
mul.wide.u16    t5, a7, b3;\n\t
mul.wide.u16    t6, a7, b5;\n\t
add.cc.u32    r1, r1, t1;\n\t
addc.cc.u32    r2, r2, t2;\n\t
addc.cc.u32    r3, r3, t3;\n\t
addc.cc.u32    r4, r4, t4;\n\t
addc.cc.u32    r5, r5, t5;\n\t
addc.cc.u32    r6, r6, t6;\n\t
addc.u32    r7, r7, 0;\n\t
mul.wide.u16    t0, a0, b1;\n\t
mul.wide.u16    t1, a0, b3;\n\t
mul.wide.u16    t2, a0, b5;\n\t
mul.wide.u16    t3, a0, b7;\n\t
mul.wide.u16    t4, a2, b7;\n\t
mul.wide.u16    t5, a4, b7;\n\t
mul.wide.u16    t6, a6, b7;\n\t
mul.wide.u16    s3, a1, b6;\n\t
add.cc.u32    t3, t3, s3;\n\t
addc.u32    t4, t4, 0;\n\t
mul.wide.u16    s3, a2, b5;\n\t
add.cc.u32    t3, t3, s3;\n\t
addc.u32    t4, t4, 0;\n\t
mul.wide.u16    s2, a1, b4;\n\t
mul.wide.u16    s3, a3, b4;\n\t
mul.wide.u16    s4, a3, b6;\n\t
add.cc.u32    t2, t2, s2;\n\t
addc.cc.u32    t3, t3, s3;\n\t
addc.cc.u32    t4, t4, s4;\n\t
addc.u32    t5, t5, 0;\n\t
mul.wide.u16    s2, a2, b3;\n\t
mul.wide.u16    s3, a4, b3;\n\t
mul.wide.u16    s4, a4, b5;\n\t
add.cc.u32    t2, t2, s2;\n\t
addc.cc.u32    t3, t3, s3;\n\t
addc.cc.u32    t4, t4, s4;\n\t
addc.u32    t5, t5, 0;\n\t
mul.wide.u16    s1, a1, b2;\n\t
mul.wide.u16    s2, a3, b2;\n\t
mul.wide.u16    s3, a5, b2;\n\t
mul.wide.u16    s4, a5, b4;\n\t
mul.wide.u16    s5, a5, b6;\n\t
add.cc.u32    t1, t1, s1;\n\t
addc.cc.u32    t2, t2, s2;\n\t
addc.cc.u32    t3, t3, s3;\n\t
addc.cc.u32    t4, t4, s4;\n\t
addc.cc.u32    t5, t5, s5;\n\t
addc.u32    t6, t6, 0;\n\t
mul.wide.u16    s1, a2, b1;\n\t
mul.wide.u16    s2, a4, b1;\n\t
mul.wide.u16    s3, a6, b1;\n\t
mul.wide.u16    s4, a6, b3;\n\t
mul.wide.u16    s5, a6, b5;\n\t
add.cc.u32    t1, t1, s1;\n\t
addc.cc.u32    t2, t2, s2;\n\t
addc.cc.u32    t3, t3, s3;\n\t
addc.cc.u32    t4, t4, s4;\n\t
addc.cc.u32    t5, t5, s5;\n\t
addc.u32    t6, t6, 0;\n\t
mul.wide.u16    s0, a1, b0;\n\t
mul.wide.u16    s1, a3, b0;\n\t
mul.wide.u16    s2, a5, b0;\n\t
mul.wide.u16    s3, a7, b0;\n\t
mul.wide.u16    s4, a7, b2;\n\t
mul.wide.u16    s5, a7, b4;\n\t
mul.wide.u16    s6, a7, b6;\n\t
add.cc.u32    t0, t0, s0;\n\t
addc.cc.u32    t1, t1, s1;\n\t
addc.cc.u32    t2, t2, s2;\n\t
addc.cc.u32    t3, t3, s3;\n\t
addc.cc.u32    t4, t4, s4;\n\t
addc.cc.u32    t5, t5, s5;\n\t
addc.cc.u32    t6, t6, s6;\n\t
addc.u32    t7, 0, 0;\n\t


//3) using warp level reduction and recursion
//---------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_warp_level_recursion(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x;
	}

    acc = warpReduceSum(acc);

    if ((threadIdx.x & (warpSize - 1)) == 0)
    {
        out_arr[blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE] = acc;
    }
}

__global__ void naive_kernel_warp_level_reduction(const ec_point* in_arr, ec_point* out_arr, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
    {   
        acc = ECC_ADD(acc, in_arr[tid]);
        tid += blockDim.x * gridDim.x;
	}

    acc = warpReduceSum(acc);

    if ((threadIdx.x & (warpSize - 1)) == 0)
    {
        out_arr[blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE] = acc;
    }
}



void naive_multiexp_warp_level_recursion_driver(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	//stage 1 : exponetiationa and reduction
    
    int blockSize1;
    int minGridSize1;
  	int realGridSize1;
	int optimalGridSize1;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize1, &blockSize1, naive_multiexp_kernel_warp_level_recursion, 0, 0);
  	realGridSize1 = (arr_len + blockSize1 - 1) / blockSize1;
	optimalGridSize1 = min(minGridSize1, realGridSize1);

	std::cout << "Grid size1: " << realGridSize1 << ",  min grid size1: " << minGridSize1 << ",  blockSize1: " << blockSize1 << std::endl;
	naive_multiexp_kernel_warp_level_recursion<<<optimalGridSize1, blockSize1>>>(point_arr, power_arr, out, arr_len);

    //stage 2 : reduction only

    int blockSize2;
    int minGridSize2;
  	int realGridSize2;
	int optimalGridSize2;

    cudaOccupancyMaxPotentialBlockSize(&minGridSize2, &blockSize2, naive_multiexp_kernel_warp_level_recursion, 0, 0);
  	realGridSize2 = (arr_len + blockSize2 - 1) / blockSize2;
	optimalGridSize2 = min(minGridSize2, realGridSize2);

    std::cout << "Grid size2: " << realGridSize2 << ",  min grid size2: " << minGridSize2 << ",  blockSize2: " << blockSize2 << std::endl;

    arr_len = optimalGridSize1 * blockSize1 / WARP_SIZE;

    while (arr_len > 1)
    {
        if (arr_len <= DEFAUL_NUM_OF_THREADS_PER_BLOCK)
        {
            naive_kernel_block_level_reduction<<<1, DEFAUL_NUM_OF_THREADS_PER_BLOCK>>>(out_arr, o)
        }
        else
        {
            /* code */
        }        
    }
}


/the main stage of Pippenger algorithm is splitting a lot op points among a relatively small amount of chunks (or bins)
//This operation can be considered as a sort of histogram construction, so we can use specific Cuda algorithms.
//Source of inspiration are: 
//https://devblogs.nvidia.com/voting-and-shuffling-optimize-atomic-operations/
//https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/

//TODO: what is the exact difference between inline and __inline__

DEVICE_FUNC __inline__ void __shfl(const ec_point& in_var, ec_point& out_var, unsigned int mask, unsigned int offset, int width=32)
{
    //ec_point = 3 * 8  = 24 int = 6 int4
    const int4* a = reinterpret_cast<const int4*>(&in_var);
    int4* b = reinterpret_cast<int4*>(&out_var);

    for (unsigned i = 0; i < 6; i++)
    {
        b[i].x = __shfl_sync(mask, a[i].x, offset, width);
        b[i].y = __shfl_sync(mask, a[i].y, offset, width);
        b[i].z = __shfl_sync(mask, a[i].z, offset, width);
        b[i].w = __shfl_sync(mask, a[i].w, offset, width);
    }
}

DEVICE_FUNC __inline__ uint32_t get_peers(uint32_t key)
{
    uint32_t peers=0;
    bool is_peer;

    // in the beginning, all lanes are available
    uint32_t unclaimed=0xffffffff;

    do
    {
        // fetch key of first unclaimed lane and compare with this key
        is_peer = (key == __shfl_sync(unclaimed, key, __ffs(unclaimed) - 1));

        // determine which lanes had a match
        peers = __ballot_sync(unclaimed, is_peer);

        // remove lanes with matching keys from the pool
        unclaimed ^= peers;


    }
    // quit if we had a match
    while (!is_peer);

    return peers;
}

DEVICE_FUNC __inline__ ec_point reduce_peers(uint peers, ec_point pt)
{
    int lane = threadIdx.x & (warpSize - 1);

    // find the peer with lowest lane index
    int first = __ffs(peers)-1;

    // calculate own relative position among peers
    int rel_pos = __popc(peers << (32 - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while(__any_sync(peers, peers))
    {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl() only works if both threads participate, so we always do.
        ec_point temp;
        __shfl(pt, temp, peers, next - 1);

        // only add if there was anything to add
        if (next)
        {
            pt = ECC_ADD(pt, temp);
        }

        // all lanes with their least significant index bit set are done
        uint32_t done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~ __ballot_sync(peers, done);

        // abuse relative position as iteration counter
        rel_pos >>= 1;
    }

    return pt;
}

// __global__ void Pippenger_multiexp_kernel(const ec_point* point_arr, const uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
// {
//     ec_point acc = point_at_infty();
    
//     size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
// 	while (tid < arr_len)
// 	{   
//         ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
//         acc = ECC_ADD(acc, x);
//         tid += blockDim.x * gridDim.x;
// 	}

//     acc = blockReduceSum(acc);
    
//     if (threadIdx.x == 0)
//         out_arr[blockIdx.x] = acc;
// }


//Pipenger: basic version - simple, yet powerful. The same version of Pippenger algorithm is implemented in libff and Bellman

struct Lock
{
    int* mutex;

    Lock()
    {
        cudaMalloc((void**)&mutex, sizeof(int));
        cudaMemset(mutex, 0, sizeof(int));
    }

    ~Lock()
    {
        cudaFree(mutex);
    }

    DEVICE_FUNC void lock()
    {
        while (atomicCAS(mutex, 0, 1) != 0);
    }

    DEVICE_FUNC void unlock()
    {
        atomicExch(mutex, 0);
    }
};

//TODO: refactor it later)

template<unsigned M>
struct LockArr
{
    int* mutex_arr;

    LockArr()
    {
        cudaMalloc((void**)&mutex_arr, sizeof(int) * M);
        cudaMemset(mutex_arr, 0, sizeof(int) * M);
    }

    ~LockArr()
    {
        cudaFree(mutex_arr);
    }

    DEVICE_FUNC void lock(size_t idx)
    {
        while (atomicCAS(mutex_arr + idx, 0, 1) != 0);
    }

    DEVICE_FUNC void unlock(size_t idx)
    {
        atomicExch(mutex_arr + idx, 0);
    }
};

p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

base_field = GF(p)
curve = EllipticCurve(GF(p), [0, 3]);
G = curve(1, 2, 1)

R = field(2 ^ 256)



\
    auto count = convert_to_non_adjacent_form(power, wnaf_arr);\
    ec_point Q = point_at_infty();\
\
    for (int j = count - 1; j >=0 ; j--)\
    {\
        auto& wnaf_data = wnaf_arr[j];\
        int8_t abs_offset;\
        bool is_negative;\
        if (wnaf_data.value >= 0)\
        {\
            abs_offset = wnaf_data.value;\
            is_negative = false;\
        }\
        else\
        {\
            abs_offset = -wnaf_data.value;\
            is_negative = true;\
        }\
\
        ec_point temp = precomputed[(abs_offset - 1)/ 2];\
        if (is_negative)\
            temp = INV(temp);\
\
        Q = ECC_ADD##SUFFIX(Q, temp);\
\
        for(uint8_t k = 0; k < wnaf_data.gap; k++)\
            Q = ECC_DOUBLE##SUFFIX(Q);\
    }\
\
   return Q;\


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