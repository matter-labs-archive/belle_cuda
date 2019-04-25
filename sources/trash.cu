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




def split(d):
    wnaf_data = []
    gap = 0
    while ( d > 0):
        if (d % 2 == 0):
            gap = gap + 1
            d = d / 2
        else:
            val = d % 16
           
            if ( val >= 8):
                val = val - 16
            
            wnaf_data.append((val, gap))
            gap = 4
            d = d - val
            if ( d % 16):
                print "error"
            d = d / 16
            
    return wnaf_data

x = 0x0c82d1d0ceb1a679195bd2308de3fcdc2085d81e72ac0b037df0ac63c18cba42
y = 0x2ddeef4b2e85fcbd491dbd10abf9b0d58c76d28ba938f4cf632dd2829f0e5f2e
z = 0x0e0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d



            
pt = curve(x, y, z)
n = 0x1e581b79b5350a43bc7f3bd165933136720a5ce0010f43bea73b5f404bf9d790



    
    
data =  split(n)[::-1]
pt_doubled = 2 * pt


precomputed.append(pt)
for _ in xrange(4):
    if _ > 0:
        precomputed.append(precomputed[_-1] + pt_doubled)

Q = curve(0, 1, 0)

for i in xrange(len(data)):
    val = data[i][0]
    gap = data[i][1]
    offset = abs(val)
    is_positive = (val >= 0)
    
    P = precomputed[(offset-1)/2]
    if is_positive:
        Q = Q + P
    else:
        Q = Q - P
        
    for j in xrange(gap):
        Q = 2 * Q

        
print Q == n * pt

x = 0x178046057c2d02cdefc68e37ee7a934d5c423e25ac2151ee3639b1b0596d4332
y = 0x2f9dbe1fd3343e0838031c8d2e8686d6dcfff58dffd8ad4d5610afd2ab76c8bd
z = 0x2ad40110c750927d33d3d5b4a13b7e03a59c7a00a5e6ef0ac622b91a447698c4


print Q == curve(x, y, z)

//Pippenger final exponentiation

__global__ void Pippenger_final_exponentiation(ec_point* in_arr, ec_point* out_arr, size_t arr_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
    {   
        ec_point pt = in_arr[tid];

        for (size_t j = 0; j < threadIdx.x; j++)
            pt = ECC_DOUBLE(pt);
        
        out_arr[tid] = pt;

        tid += blockDim.x * gridDim.x;
	}
}

__global__ void multiexp_Pippenger(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len, int* mutex_arr)
{
    ec_point acc = point_at_infty();
    
    size_t start = (arr_len / gridDim.x) * blockIdx.x;
    size_t end = (arr_len / gridDim.x) * (blockIdx.x + 1);

    for (size_t i = start; i < end; i++)
    {
        if (get_bit(power_arr[i], threadIdx.x))
            acc = ECC_MIXED_ADD(acc, point_arr[i]);
    }

    while (atomicCAS(mutex_arr + threadIdx.x, 0, 1) != 0);
    out_arr[threadIdx.x] = ECC_ADD(out_arr[threadIdx.x], acc);
    atomicExch(mutex_arr + threadIdx.x, 0);   
}

void Pippenger_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;

    size_t M = 256;
    int* mutex_arr;
    cudaMalloc((void**)&mutex_arr, sizeof(int) * M);
    cudaMemset(mutex_arr, 0, sizeof(int) * M);

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, multiexp_Pippenger, 0, 4 * N * 3 * WARP_SIZE);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;;

    //but here we need an array of such elements!

    ec_point point_at_infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };

    for (size_t j = 0 ; j < 256; j++)
    {
        cudaMemcpy(out_arr + j, &point_at_infty, sizeof(ec_point), cudaMemcpyHostToDevice);
    }
    
	std::cout << "Real grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;

	multiexp_Pippenger<<<realGridSize, 256>>>(point_arr, power_arr, out_arr, arr_len, mutex_arr);
    cudaDeviceSynchronize();

    Pippenger_final_exponentiation<<<1, 256>>>(out_arr, out_arr, 256);
    cudaDeviceSynchronize();

    naive_kernel_block_level_reduction<<<1, 256>>>(out_arr, out_arr, 256);

    cudaFree(mutex_arr);
}


//Bitonic Sort
//------------------------------------------------------------------------------------------------------------------------------------------
//TODO: may be better and simpler to use CUB library? (it contains primitives for sorting)


#define LARGE_C 8
#define LARGE_CHUNK_SIZE ()

__global__ void bitonic_sort_step(ec_point* values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}



void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

//scan of large arrays :()


//third step - reduction of rolling sum


//This is a version of Pippenger algorithm with large bins
//------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------

#define LARGE_C 16
#define LARGE_CHUNK_SIZE 256

__global__ void bitonic_sort_step(ec_point* values, int j, int k)
{
    /* Sorting partners: i and ixj */
    unsigned int i, ixj; 
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if (ixj > i)
    {
        if ( (i & k ) == 0 )
        {
            /* Sort ascending */
            if (dev_values[i]>dev_values[ixj])
            {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i&k)!=0)
        {
            /* Sort descending */
            if (dev_values[i]<dev_values[ixj])
            {
                /* exchange(i,ixj); */
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

void bitonic_sort(ec_point* values)
{
    float *dev_values;
    size_t size = NUM_VALS * sizeof(float);

    cudaMalloc((void**) &dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */

    int j, k;
    /* Major step */
    for (k = 2; k <= NUM_VALS; k <<= 1)
    {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1)
        {
            bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        }
    }
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

__global__ void scan_and_reduce(const ec_point* global_in_arr, ec_point* out)
{
    // allocated on invocation
    extern __shared__ ec_point temp[];

    //scanning

    uint tid = threadIdx.x;
    uint offset = 1;
    const ec_point* in_arr = global_in_arr + blockIdx.x * SMALL_CHUNK_SIZE;

    uint ai = tid;
    uint bi = tid + (SMALL_CHUNK_SIZE / 2);

    uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint bankOffsetB = CONFLICT_FREE_OFFSET(ai);
    temp[ai + bankOffsetA] = in_arr[ai];
    temp[bi + bankOffsetB] = in_arr[bi]; 

    // build sum in place up the tree
    for (int d = SMALL_CHUNK_SIZE >> 1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset*(2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 

            temp[bi] = ECC_ADD(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    
    if (tid == 0)
    {
        temp[SMALL_CHUNK_SIZE - 1 + CONFLICT_FREE_OFFSET(SMALL_CHUNK_SIZE - 1)] = point_at_infty();
    }

    // traverse down tree & build scan
    for (uint d = 1; d < SMALL_CHUNK_SIZE; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 
            
            ec_point t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = ECC_ADD(t, temp[bi]);
        }
    }
    
    __syncthreads();
   
    //reducing

    for (int d = SMALL_CHUNK_SIZE >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
            temp[tid] = temp[tid + d];
        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x * SMALL_CHUNK_SIZE] = temp[0];
}


p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

base_field = GF(p)
curve = EllipticCurve(base_field, [0, 3]);
G = curve(1, 2, 1)

R = base_field(2 ^ 256)

def to_mont_form(x):
    return x * R

def from_mont_form(x):
    return x / R

def parse_affine_point(line1, line2):
    x = int(line1.split('=')[1], 0x10)
    y = int(line2.split('=')[1], 0x10)
    
    return curve(x, y, R)

def parse_bignum(line, base = 0x10):
    return int(line, base)

def parse_ec_point(line1, line2, line3):
    x = int(line1.split('=')[1], 0x10)
    y = int(line2.split('=')[1], 0x10)
    z = int(line3.split('=')[1], 0x10)
    
    return curve(x, y, z)

A_arr = []
B_arr = []
C_arr = []

FILE_LOCATION = "/home/k/TestCuda3/benches.txt"

file = open(FILE_LOCATION, "r")

bench_len = parse_bignum(file.readline().split("=")[1][:-1], 10)
print bench_len
print "Start!"

print file.readline()

for _ in xrange(bench_len):
    x = file.readline()[:-1]
    y = file.readline()[:-1]
    file.readline()
    
    A_arr.append(parse_affine_point(x, y))
    
print file.readline()
print file.readline()

for _ in xrange(bench_len):
    num = file.readline()[:-1]
    B_arr.append(parse_bignum(num))
    
print file.readline()
print file.readline()

x = file.readline()[:-1]
y = file.readline()[:-1]
z = file.readline()[:-1]
file.readline()
    
C = parse_ec_point(x, y, z)

# x = file.readline()[:-1]
# y = file.readline()[:-1]
# z = file.readline()[:-1]
# file.readline()

# C += parse_ec_point(x, y, z)

# x = file.readline()[:-1]
# y = file.readline()[:-1]
# z = file.readline()[:-1]
# file.readline()

# C += parse_ec_point(x, y, z)

acc = curve(0, 1, 0)

for i in xrange(bench_len):
    acc += B_arr[i] * A_arr[i]
    
print acc == C
    







p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

base_field = GF(p)
curve = EllipticCurve(base_field, [0, 3]);
G = curve(1, 2, 1)

R = base_field(2 ^ 256)

def to_mont_form(x):
    return x * R

def from_mont_form(x):
    return x / R

def parse_affine_point(line1, line2):
    x = int(line1.split('=')[1], 0x10)
    y = int(line2.split('=')[1], 0x10)
    
    return curve(x, y, R)

def parse_bignum(line, base = 0x10):
    return int(line, base)

def parse_ec_point(line1, line2, line3):
    x = int(line1.split('=')[1], 0x10)
    y = int(line2.split('=')[1], 0x10)
    z = int(line3.split('=')[1], 0x10)
    
    return curve(x, y, z)


A_arr = []
B_arr = []
C_arr = []

FILE_LOCATION = "/home/k/TestCuda3/benches.txt"

file = open(FILE_LOCATION, "r")

bench_len = parse_bignum(file.readline().split("=")[1][:-1], 10)
print bench_len
print "Start!"

print file.readline()

for _ in xrange(bench_len):
    x = file.readline()[:-1]
    y = file.readline()[:-1]
    file.readline()
    
    A_arr.append(parse_affine_point(x, y))
    
print file.readline()
print file.readline()

for _ in xrange(bench_len):
    num = file.readline()[:-1]
    B_arr.append(parse_bignum(num))
    
print file.readline()
print file.readline()

x = file.readline()[:-1]
y = file.readline()[:-1]
z = file.readline()[:-1]
file.readline()
    
C = parse_ec_point(x, y, z)

acc = curve(0, 1, 0)

for j in xrange(bench_len):
    acc += (B_arr[j]) * A_arr[j]
    
print acc == C

        
# def extractKBits(num,k,p): 
  
#     num = num >> p
#     num = num & (2^k - 1)
#     return num
    
# for i in xrange(32):

#     x = file.readline()[:-1]
#     y = file.readline()[:-1]
#     z = file.readline()[:-1]
#     file.readline()
    
#     C = parse_ec_point(x, y, z)

#     acc = curve(0, 1, 0)

#     for j in xrange(bench_len):
#         acc += extractKBits(B_arr[j], 8, i * 8 ) * A_arr[j]
   
#     print acc == C

ec_point& incr = incr_arr[]
        for (j = 0; j < SCAN_BLOCK_SIZE; j++)
        {
           
            [tid * SCAN_BLOCK_SIZE + j] = E
            acc = ECC_ADD(acc, in_arr[tid]);
        tid += blockDim.x * gridDim.x;
        }


//Some experiements with CUB library
//----------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

//Do not compile: may be I should ask on stackoverflow?

#ifdef OLOLO_TROLOLO

#include <cub/cub.cuh>   

struct ec_point_adder_t
{
    DEVICE_FUNC __forceinline__
    ec_point operator()(const ec_point& a, const ec_point& b) const
    {
        return ECC_ADD(a, b);
    }
};

__global__ void exp_componentwise(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        out[tid] = ECC_EXP(point_arr[tid], power_arr[tid]);
        tid += blockDim.x * gridDim.x;
	} 
}

void CUB_reduce_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;
    int maxActiveBlocks;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, exp_componentwise, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, exp_componentwise, blockSize, 0);

    realGridSize = (arr_len + blockSize - 1) / blockSize;
    realGridSize = min(realGridSize, maxActiveBlocks * smCount);
      
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    ec_point_adder_t ec_point_adder;

     ec_point infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };


    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, out_arr, out_arr, arr_len, ec_point_adder, infty);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run reduction
    //cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,  out_arr, out_arr, arr_len, ec_point_adder, point_at_infty());

    cudaFree(d_temp_storage);
}

#endif


p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
r = 21888242871839275222246405745257275088548364400416034343698204186575808495617

base_field = GF(p)
curve = EllipticCurve(base_field, [0, 3]);
G = curve(1, 2, 1)

R = base_field(2 ^ 256)

def to_mont_form(x):
    return x * R

def from_mont_form(x):
    return x / R


def mont_inv(x):
    a = from_mont_form(x) ^ (-1)
    return to_mont_form(a)

def mont_mul(a, b):
    return a * b / R

def Kasinski_algo(x):
    U = p
    V = x
    R = 0
    S = 1
    k = 0
    while V > 0:
        if (U % 2 == 0):
            U = U / 2
            S = 2 * S
        elif (V % 2 == 0):
            V = V / 2
            R = 2 * R
        elif (U > V):
            U = (U - V ) / 2
            R = R + S
            S = 2 * S
        else:
            V = (V - U ) / 2
            S = S + R
            R = 2 * R
        k = k +1
    if (R >= p):
        R = R - p
    return (p - R, k)
    
    
x = 0x001eccd03356f7b8e40123519a541539c16e68ebf437d1a1e9ad2977aba01e60


a = 0x2e23e7a5433d5e3285f09bc1a8906294407a94c2b09eb52fc15985eeff121223

print base_field(a) == mont_inv(x)
elem, k = Kasinski_algo(x)

print "elem: ", hex(elem)
print "k :", k

y = base_field(elem)
z = base_field(2 ^ (512 - k))

q = mont_mul(y, z)
print hex(int(q))
print mont_mul(q, R^2) == mont_inv(x)
print hex(int(mont_mul(q, R^2)))

print 512 - 354
print int(158 / 32)
print 158 % 32


x = 0x2762fc49f4227b8dc9a1abcd8a31d83b536d7590a3fcfd17910e6a45093112f4
y = 0x2970e1f1c83d1bc0146bb7c7a176c3773050dfb1620fd2d367b326a31e56e274
z = 0x1b8bd2d05fe5a2a4c86cb8b7b18e532a907bb0895dac2a62c4766623d46bee47

P = curve(x, y, z)

x = 0x1070b11dcf4e9e53efa929263a959a2e0692d26a46d75f9caea67eea2dce74c2
y = 0x22f2c80cee23c883f966c2091f878a1d99718b2de3234fb9102ccd13a8d9703f
z = 0x10b36d77cc648a20dd2666035eeb81bce32810bfd2519b136061163e0f6ca90d

Q = curve(x, y, z)

print P == Q
print "p"

MAX_POWER_BITLEN = 256
LARGE_C = 16
LARGE_CHUNK_SIZE = 2 ^ 16


NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;

ARR_LEN = NUM_OF_CHUNKS * LARGE_CHUNK_SIZE

print NUM_OF_CHUNKS
print ARR_LEN

for(int32_t thread=blockIdx.x*blockDim.x+threadIdx.x;thread<count;thread+=blockDim.x*gridDim.x) {
    int aindex=thread, bindex=thread, outindex=thread;
    if(NULL!=a_indices) aindex=a_indices[thread%a_indices_count];
    if(NULL!=b_indices) bindex=b_indices[thread%b_indices_count];
    if(NULL!=out_indices) outindex=out_indices[thread];

    data=a_data + aindex%a_count;
    #pragma unroll
    for(int index=0;index<a_size;index++) {
      if(index<a_len)
        A[index]=data[index*a_stride];
      else
        A[index]=0;
    }

    data=b_data + bindex%b_count;
    #pragma unroll
    for(int index=0;index<b_size;index++) {
      if(index<b_len)
        B[index]=data[index*b_stride];
      else
        B[index]=0;
    }

    mul(P, A, B);

    data=p_data + outindex;
    #pragma unroll
    for(int index=0;index<a_size+b_size;index++)
      if(index<p_len)
        data[index*p_stride]=P[index];

    for(int index=a_size+b_size;index<p_len;index++)
      data[index*p_stride]=0;



DEVICE_FUNC void add_uint512_in_place_asm(uint512_g& lhs, const uint512_g& rhs)
{
	asm (	"add.cc.u32      %0,  %0,  %16;\n\t"
         	"addc.cc.u32     %1,  %1,  %17;\n\t"
            "addc.cc.u32     %2,  %2,  %18;\n\t"
            "addc.cc.u32     %3,  %3,  %19;\n\t"
            "addc.cc.u32     %4,  %4,  %20;\n\t"
            "addc.cc.u32     %5,  %5,  %21;\n\t"
            "addc.cc.u32     %6,  %6,  %22;\n\t"
            "addc.u32        %7,  %7,  %23;\n\t"
            "add.cc.u32      %8,  %8,  %24;\n\t"
         	"addc.cc.u32     %9,  %9,  %25;\n\t"
            "addc.cc.u32     %10, %10, %26;\n\t"
            "addc.cc.u32     %11, %11, %27;\n\t"
            "addc.cc.u32     %12, %12, %28;\n\t"
            "addc.cc.u32     %13, %13, %29;\n\t"
            "addc.cc.u32     %14, %14, %30;\n\t"
            "addc.u32        %15, %15, %31;\n\t"
            :   "+r"(lhs.n[0]), "+r"(lhs.n[1]), "+r"(lhs.n[2]), "+r"(lhs.n[3]), "+r"(lhs.n[4]), "+r"(lhs.n[5]), "+r"(lhs.n[6]), "+r"(lhs.n[7]),
				"+r"(lhs.n[8]), "+r"(lhs.n[9]), "+r"(lhs.n[10]), "+r"(lhs.n[11]), "+r"(lhs.n[12]), "+r"(lhs.n[13]), "+r"(lhs.n[14]), "+r"(lhs.n[15])        		
			:   "r"(rhs.n[0]), "r"(rhs.n[1]), "r"(rhs.n[2]), "r"(rhs.n[3]), "r"(rhs.n[4]), "r"(rhs.n[5]), "r"(rhs.n[6]), "r"(rhs.n[7]),
				"r"(rhs.n[8]), "r"(rhs.n[9]), "r"(rhs.n[10]), "r"(rhs.n[11]), "r"(rhs.n[12]), "r"(rhs.n[13]), "r"(rhs.n[14]), "r"(rhs.n[15]));
}

/--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------

//warp-based elliptic curve point addition

//16 threads are used to calculate one operation

#define THREADS_PER_ECC_ADD 16

DEVICE_FUNC __inline__ bool is_leader_lane()
{
    return (threadIdx.x % THREADS_PER_ECC_ADD == 0);
}

//chose an element based on subwarp

DEVICE_FUNC __inline__ uint32_t subwarp_chooser(const uint256_g& X, const uint256_g& Y, uint32_t lane, bool rrs)
{
    uint256& temp = (rrs ? X : Y);
    return temp.n[lane];
}

DEVICE_FUNC __inline__ uint32_t subwarp_chooser(uint32_t X, uint32_t Y, bool rrs)
{
    return (rrs ? X : Y);
}

DEVICE_FUNC __inline__ uint32_t warp_based_field_sub(uint32_t A, uint32_t B, uint32_t mask, uint32_t warp_idx, uint32_t lane)
{
    if (warp_based_geq(A, B, mask, warp_idx))
    {
        return warp_based_sub(A, B, mask);
    }
    else
    {
        uint32_t temp = warp_based_add(A, BASE_FIELD_P.n[lane]);
        return warp_based_sub(temp, B, mask);
    }
}

DEVICE_FUNC __inline__ uint32_t warp_based_exchanger(uint32_t A, uint32_t B, bool rrs)
{
    uint32_t elem = (rrs ? A : B);
    return __shfl_down_sync(0xFFFFFFFF, elem, THREADS_PER_MUL, THREADS_PER_ECC_ADD);
}

DEVICE_FUNC void ECC_add_proj_warp_based(const ec_point& left, const ec_point& right, ec_point& OUT)
{
    uint32_t exit_flag = 0;
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t warp_idx = tid / THREADS_PER_MUL;
    uint32_t mask = (THREADS_PER_MUL - 1) << (warp_idx * THREADS_PER_MUL);
    uint32_t lane = tid % THREADS_PER_MUL;
    bool rrs = (warp_idx % THREADS_PER_ECC_ADD) < (THREADS_PER_ECC_ADD / 2 );

    if (is_leader_lane())
    {
        if (is_infinity(A))
        {
            C = B;
            exit_flag = 1;
        }
	    else if (is_infinity(B))
        {
		    C = A;
            exit_flag = 1;
        }
    }

    exit_flag = __shfl_sync(0xFFFFFFFF, exit_flag, 0, THREADS_PER_ECC_ADD);
    if (exit_flag)
        return;

    uint32_t Z = subwarp_chooser(left.z, right.z, lane, rrs);
    uint32_t Y = subwarp_chooser(right.y, left.y);
    uint32_t U12 = mont_mul_warp_based(Z, Y);

    uint32_t X = subwarp_chooser(right.x, left.x);
    uint32_t V12 = mont_mul_warp_based(Z, X);

    uint32_t U_V = subwarp_chooser(U12, V12, rrs);
    uint32_t temp = warp_based_exchanger(V12, U12, rrs);

    U_V = warp_based_field_sub(U_V, temp, mask, warp_idx, lane);
    

    //check for equality

    //squaring of U and V:
    uint32_t U_V_sq = mont_mul_warp_based(U_V, U_V);

    temp = warp_based_exchanger(V12, U12, rrs);
   
	uint256_g Vcube = MONT_MUL(Vsq, V);
	uint256_g W = MONT_MUL(left.z, right.z);

	temp1 = MONT_MUL(temp1, W);
    temp2 = MONT_MUL(BASE_FIELD_R2, Vsq);

    temp2 = MONT_MUL(temp2, V2);
    res.z = MONT_MUL(Vcube, W);

    tempx = MONT_MUL(Vsq, V2);
    temp3 = MONT_MUL(Vcube, U2);

    //without pair
    temp1 = FIELD_SUB(temp1, Vcube);
    uint256_g A = FIELD_SUB(temp1, temp2);
    tempx = FIELD_SUB(tempx, A);

	res.x = MONT_MUL(V, A);	
	tempx = MONT_MUL(U, tempx);
	
	res.y = FIELD_SUB(tempx, temp3);

	
	return res;

}

__global__ void ECC_add_proj_warp_based_kernel(const ec_point* a_arr, const ec_point* b_arr, ec_point *c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x);
    size_t idx = tid / THREADS_PER_ECC_ADD;
	while (idx < arr_len)
	{
		ECC_add_proj_warp_based(a_arr[idx], b_arr[idx], c_arr[idx]);
		tid += (blockDim.x * gridDim.x) / THREADS_PER_ECC_ADD;
	}
}

void ECC_add_proj_warp_based_driver(const ec_point* a_arr, const ec_point* b_arr, ec_point* c_arr, size_t arr_len)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t smCount = prop.multiProcessorCount;   
    geometry2 geometry = find_geometry2(ECC_add_proj_warp_based_kernel, 0, uint32_t smCount);

    std::cout << "Grid size: " << geometry.gridSize << ",  blockSize: " << geometry.blockSize << std::endl;
    ECC_add_proj_warp_based_kernel<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}


#define UNROLLED_CYCLE_ITER(idx) \
"mov.b32 sp, b;\n\t" \
"mov.b32 x, sp;\n\t" 

// "mad.lo.cc.u32 v, a, x, v;\n\t" \
// "madc.hi.cc.u32 u, a, x, u;\n\t" \
// "addc.u32 t, 0, 0;\n\t" \
// "mov.b32 sp, v;\n\t" \
// "shfl.sync.down.b32 sp, sp, 1, 8;\n\t" \
// "mov.b32 v, sp;\n\t" \
// "mov.b32 sp, c;\n\t" \
// "mov.b32 c, sp;\n\t" \
// "and.type y, %laneid, 7;\n\t" \
// "setp.eq.u32  p, y, 7;\n\t"\
// "@p {\n\t" \
// "mov.u32 c, v;\n\t" \
// "mov.u32 v, 0;\n\t" \
// "}\n\t" \
// "add.cc.u32 v, u, v;\n\t" \
// "addc.u32 u, t, 0;\n\t"

//"shfl.sync.idx.b32 sp, sp, idx, 8;" \
//"shfl.sync.up.b32 sp, sp, 1, 8;\n\t" \
//"shfl.sync.up.b32  sp, sp, 1, 8;\n\t" \

//The following implementation is based on paper "A Warp-synchronous Implementation for Multiple-length Multiplication on the GPU"

DEVICE_FUNC uint64_g asm_mul_warp_based(uint32_t A, const uint32_t B)
{
    uint64_g res;
    
    asm(    "{\n\t"  
            ".reg .u32 a, b, x, y, u, v, c, t;\n\t"
            ".reg .b32 sp, sp1;\n\t"
            ".reg .pred p;\n\t"

            "mov.u32 a, %1;\n\t"
            "mov.u32 b, %2;\n\t"

            "mov.u32 u, 0;\n\t"
            "mov.u32 v, 0;\n\t"
            "mov.u32 c, 0;\n\t"

            // UNROLLED_CYCLE_ITER(0)
            // UNROLLED_CYCLE_ITER(1)
            // UNROLLED_CYCLE_ITER(2)
            // UNROLLED_CYCLE_ITER(3)
            // UNROLLED_CYCLE_ITER(4)
            // UNROLLED_CYCLE_ITER(5)
            // UNROLLED_CYCLE_ITER(6)
            // UNROLLED_CYCLE_ITER(7)

            "L1:\n\t"
            "setp.eq.u32 p, u, 0;\n\t"
            "vote.sync.any.pred  p, p, 8;\n\t"
            "@!p bra L2;\n\t"
            "mov.b32 sp, u;\n\t"
            "shfl.sync.up.b32  sp1, sp, 0x1,  0x0, 0xffffffff;\n\t"
            //"shfl.sync.down.b32 sp1, sp, 1, 0xffffffff;\n\t"
            // "mov.b32 u, sp;\n\t"
            // "add.cc.u32 v, v, u;\n\t"
            // "addc.u32 u, 0, 0;\n\t"
            // "bra L1;\n\t"
           
            "L2:\n\t"

            // "st.global.u32 [OUT + %laneid], c;\n\t"
            // "st.global.u32 [OUT + %laneid + 8], v;\n\t"
            "mov.b64 %0, {c, v};}\n\t"  
            : "=l"(res.as_long) : "r"(A), "r"(B));
    
    return res;
}



__device__ void d_mul_concurrent(P_BASE __restrict A , P_BASE __restrict B , P_BASE __restrict R , BASE n,
    unsigned int r , unsigned int d)
{
    // Shared conditional
    __shared__ int cond[33];

    // Result will be stored here
    unsigned long long u = 0;
    // Cache locally
    unsigned long long a = A[IDX];
    BASE b = B[IDX];
    int index = IDX / r;
    int base = index * r;
    int up = mod(IDX + 1,r) + base;

    // Perform operations
    for( unsigned int i = 0 ; i < r ; i++ )
    {
        TMP[IDX] = b;
        // Do the computation
        u = TMP[base + i] * a + u;
        // save the carry
        TMP[IDX] = u; 

        unsigned long long overflow = hi( u );
        u = (BASE)u + ( TMP[base] * d) * (unsigned long long)n;
        TMP[IDX] = u;
        overflow += hi(u);
        u = overflow + TMP[up] * ( mod(IDX , r) != (r - 1) );
    }

    index++;
    base = mod(IDX , r) > 0;

    // Propagate all carries (Rarely happens)
    do
    {
cond[index] = false;
TMP[up] = hi( u );
u = (BASE)u + TMP[IDX] * base;
if( hi(u) ) cond[index] = true;
// Stall
}while( cond[index] );
for( int i = r - 1 ; i >= 0 ; i-- ){
cond[index * (base == i)] = (u - n);
if( cond[index] < 0 ) {
break;
}
if( cond[index] ) {
u = u + (BASE)(~n) + 1 - base;
do{
cond[index] = false;
TMP[up] = hi( u );
u = (BASE)u + TMP[IDX] * base;
if( hi(u) ) cond[index] = true;
} while( cond[index] );
break;
}
}
// Copy result to global memory
R[IDX] = u;
}



import math

EMBD_DEGREES = [12]
START_DISCR = -3
STATIC_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def get_embedding_degree(r):
    for k in EMBD_DEGREES:
        if (r - 1) % k == 0:
            return k
    return 0


def get_root_of_unity(r, k):
    field = GF(r)
    gen = field.multiplicative_generator()
    a = (r - 1) / (k)
    root_of_unity = gen ^ a
    return root_of_unity

    
def get_params(r, k):
    D = START_DISCR
    field = GF(r)
    
    while True:
        if kronecker(-D, r) != 1:
            D -= 1
            continue
            
        g = get_root_of_unity(r, k)
        t_ = g + 1
        u_ = (t_ - 2) / field(-D).sqrt()
    
        t = int(t_)
        u = int(u_)
        p = (t^2 + D * u^2) / 4
        print p
        if p in ZZ and p in Primes():
            return D, p, g
        D -=1

    
def get_curve_params(D):
    if D == -4:
        return (-1, 0)
    if D== -3:
        return (0, -1)
    
    db = HilbertClassPolynomialDatabase()
    Hilbert_poly = db(D)
    coeffs = Hilbert_poly.coefficients()
    
    R = PolynomialRing(field,'x')
    reduced_poly = sum([field(b) * x^a for a,b in enumerate(coeffs)])
    j = reduced.roots()[0]
    
    c = j/(j - field(1728))
    r = field(-3)*c
    s = field(2)*c
    
    return (r, s)


def Cocks_Pinch(r):
    k = get_embedding_degree(r)
    if k == 0:
        return False
    
    D, p, g = get_params(r, k)
    A, B = get_curve_params(D)
    
    base_field = GF(p)
    extension_field = GF(p^k, name = 't')
    ext_field_modulus = extension_field.modulus()
    

if __name__ == "__main__":
    Cocks_Pinch(STATIC_R) 
    
 //-------------------------------------------------------------------------------------------------------------------------------------------------

 import math

EMBD_DEGREES = [6, 12, 24]
START_DISCR = -3
STATIC_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def check_embedding_degree(r):
    for k in EMBD_DEGREES:
        if (r - 1) % k == 0:
            return True
    return False


def get_root_of_unity(r, k):
    field = GF(r)
    gen = field.multiplicative_generator()
    a = (r - 1) / (k)
    root_of_unity = gen ^ a
    return root_of_unity

    
def get_params(r):
    D = START_DISCR
    field = GF(r)
    
    while True:
        if kronecker(-D, r) != 1:
            D -= 1
            continue
        
        for k in EMBD_DEGREES:
            if (r - 1) % k == 0:
                g = get_root_of_unity(r, k)
                t_ = g + 1
                u_ = (t_ - 2) / field(-D).sqrt()
    
                t = int(t_)
                u = int(u_)
                p = (t^2 + D * u^2) / 4

                if p in ZZ and p in Primes():
                    return D, p, g
        D -=1

    
def get_curve_params(D):
    if D == -4:
        return (-1, 0)
    if D== -3:
        return (0, -1)
    
    db = HilbertClassPolynomialDatabase()
    Hilbert_poly = db(D)
    coeffs = Hilbert_poly.coefficients()
    
    R = PolynomialRing(field,'x')
    reduced_poly = sum([field(b) * x^a for a,b in enumerate(coeffs)])
    j = reduced.roots()[0]
    
    c = j/(j - field(1728))
    r = field(-3)*c
    s = field(2)*c
    
    return (r, s)


def Cocks_Pinch(r):
    if not check_embedding_degree(r):
        print "Unsatisfiable"
        return False

    D, p, g = get_params(r)
    print p, D, k
    A, B = get_curve_params(D)
    print A, B
    
    base_field = GF(p)
    extension_field = GF(p^k, name = 't')
    ext_field_modulus = extension_field.modulus()
    

    

if __name__ == "__main__":
    Cocks_Pinch(STATIC_R) 




//-----------------------------------------------------------------------------------------------------------------------------------------------------
import math

EMBD_DEGREES = [6, 12, 24]
START_DISCR = -3
STATIC_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def check_embedding_degree(r):
    for k in EMBD_DEGREES:
        if (r - 1) % k == 0:
            return True
    return False


def get_root_of_unity(r, k):
    field = GF(r)
    gen = field.multiplicative_generator()
    a = (r - 1) / (k)
    root_of_unity = gen ^ a
    return root_of_unity

    
def get_params(r):
    D = START_DISCR
    field = GF(r)
    
    while True:
        if kronecker(-D, r) != 1:
            D -= 1
            continue
        
        for k in EMBD_DEGREES:
            if (r - 1) % k == 0:
                g = get_root_of_unity(r, k)
                t_ = g + 1
                u_ = (t_ - 2) / field(-D).sqrt()
    
                t = int(t_)
                u = int(u_)
                p = (t^2 + D * u^2) / 4

                if p in ZZ and p in Primes():
                    return D, p, g
        D -=1

    
def get_curve_params(D):
    if D == -4:
        return (-1, 0)
    if D== -3:
        return (0, -1)
    
    db = HilbertClassPolynomialDatabase()
    Hilbert_poly = db(D)
    coeffs = Hilbert_poly.coefficients()
    
    R = PolynomialRing(field,'x')
    reduced_poly = sum([field(b) * x^a for a,b in enumerate(coeffs)])
    j = reduced_poly.roots()[0]
    
    c = j/(j - field(1728))
    r = field(-3)*c
    s = field(2)*c
    
    return (r, s)


def Cocks_Pinch(r):
    if not check_embedding_degree(r):
        print "Unsatisfiable"
        return False

    D, p, g = get_params(r)
    print p, D, k
    A, B = get_curve_params(D)
    print A, B
    
    base_field = GF(p)
    extension_field = GF(p^k, name = 't')
    ext_field_modulus = extension_field.modulus()
    

    

if __name__ == "__main__":
    Cocks_Pinch(STATIC_R)


#FFT

p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
field = GF(p)
R = field(0xe0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb)
root_of_unity = field(0x1860ef942963f9e756452ac01eb203d8a22bf3742445ffd6636e735580d13d9c) / R

unity_order = root_of_unity.multiplicative_order()


# def DFT(arr):
#     n = len(arr)
#     omega = root_of_unity ^ ()

#FFT

p = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
field = GF(p)
R = field(0xe0a77c19a07df2f666ea36f7879462e36fc76959f60cd29ac96341c4ffffffb)
root_of_unity = field(0x1860ef942963f9e756452ac01eb203d8a22bf3742445ffd6636e735580d13d9c) / R
FILE_LOCATION = "/home/k/TestCuda3/benches.txt"

unity_order = root_of_unity.multiplicative_order()

#all elements of arr are given in Montgomery form

def from_mont_form(arr):
    return  [x / R for x in arr]


def to_mont_form(arr):
    return [x * R for x in arr]


def DFT(arr):
    n = len(arr)
    omega = root_of_unity ^ (unity_order / n)
    res = []
    
    temp_arr = from_mont_form(arr)
    
    
    for i in xrange(n):
        temp = field(0)
        for j, elem in enumerate(temp_arr):
            temp += elem * omega ^ (i * j)
        res.append(temp)
    
    return to_mont_form(res)
    #return [hex(int(x)) for x in to_mont_form(res)]


def parse_bignum(line, base = 0x10):
    return int(line, base)


def read_arr(bench_len, file):
    arr = []
    
    file.readline()
    file.readline()

    for _ in xrange(bench_len):
        line = file.readline()
        num = parse_bignum(line)
        arr.append(field(num))
        
    return arr

    

def sample_from_file():
    file = open(FILE_LOCATION, "r")

    bench_len = parse_bignum(file.readline().split("=")[1][:-1], 10)
    print bench_len
    
    A = read_arr(bench_len, file)
    B = read_arr(bench_len, file)
    C = read_arr(bench_len, file)
    
    D = DFT(A)
    for i in xrange(bench_len):
        if D[i] != C[i]:
            print "arrs are different"
            
    print "finish"
    

#sample_from_file()

p1 = 0x083b1dd3d6c0843fae9bb09e95158f5764bf9cbb0515bf9143f5cb2b7f1536fe
p2 = 0x0ba2e900be028f16bd311017dc9545f2f74d31477ad8bfd7674dd4d3a2a069eb
p3 = 0x14a5afe597f52ae505ed334eebedb320acf0c4cdf2be23e35c6a8b2dcfa13e45
p4 = 0x1c6175902c4a3213c080aa86e2782ee4fbcd6894e323786a821deb245eb7d736
p5 = 0x1dd4d3fc06ff0011014150f55a40a020886d509dfb6a491b938dad292f774b95
p6 = 0x0195e7aef1e7d2d5ba5524798cf111e79fdda4cd0756b1b5f380445d6508353e
p7 = 0x03895f2b40615fef7f70267a8fea0c3debf72cc34637f6f9dc61853f7c2e1ac4
p8 = 0x1a4c045266b5a98ce3d997b8b66c7fd9e0eb4ddc0e02503aebe7385c9c859711
p9 = 0x0ed73d9f62f863d0f78a292520efa0995f9e290b638a7b219127e38b03e1f570
p10 = 0x0be73162cb2408f7ce004222d6279cb4d670e97788c30fe79b9be6d3bb511961
p11 = 0x113c473bd3461acf1cfb2e051c6e3b06e9601a193ac12537370a848921091fcb
p12 = 0x13e56fe6d060d52102f41a3b6b540d2e6491c330fc5c118e59d7b9b053232e98
p13 = 0x18aaa2325496583e861dfbb5b128555e6efd8ff1580ae7276b55e84da11b512b
p14 = 0x11d49f7050c55e869ac5c2999e4d65ddb959a071b583ad811c1395e17b73a298
p15 = 0x136b14d52d7486b632c666b74c409f3fbb05f0c315ab1a79b333c12f4997840e
p16 = 0x116200968e369020e7d826b7cdd3069f170684e0d2fe4fb6c90e335049856c5f
p17 = 0x19ee1aafeb1f58d688ae355f38b2493be721f6a332821cb963bfe1233bb4c522
p18 = 0x07b3b1d63b3d430d55e4c788b12feede3166a78fc2fb017f35c55cebc3359daf
p19 = 0x12ab9035e6083e722b1018753df65d7c36f4d7a5c1f0cb7b344b9635cfb52361
p20 = 0x08bc5940d63ed30efad5680b9048e6f0987fdb5c38cee6722f27e1bcb1e360f2
p21 = 0x010a5ac63464d27e3d7a86449592f88c42a051fd161f1b7b007e6669a0c22131
p22 = 0x106d3c230f7cd9a296cdf1f4f4f7725b97b0661d89330b05ad2b25adabde651a
p23 = 0x1d61df3f70325a6ca9edaddda678aee9ce5f20eba366d5a189b18177cd759eb0
p24 = 0x05971469512c5551f5f1bc158b78dc4d1b39a4c44ff404626ff0e754833db4bd
p25 = 0x15019cdbb8d14a9de21c9ae4fdcb263579507fe34891607942b8d2537e20297c
p26 = 0x11c06a331ac685f49636b7b19b814bd8de11efafed150dbf1651d33b4feef44d
p27 = 0x06d5c14eb97dcd161d223e96364e39fac6cca3e9ace0ea073718925974e153cf
p28 = 0x074e5513fd06baf21018bc7e680771623d1459805493c7dedb718000e819297c
p29 = 0x179561276abe2c0144c95316a43bf7a279df81f9c42947aff70b1265e8e30a7f
p30 = 0x0f464d8b562b04657c07ab6a5c40beb98360c9b967a98569512ccf394ec0199c
p31 = 0x1dae8fc4b1c0e0095e298d04764788935ae648abf2e1c2e1c00511270e1eb992
p32 = 0x0d0a3e5a6e75642bca03c6ba3939520bd4a47d308400020667952ba0923f9443

arr = DFT([field(p1), field(p2), field(p3), field(p4), field(p5), field(p6), field(p7), field(p8),
          field(p9), field(p10), field(p11), field(p12), field(p13), field(p14), field(p15), field(p16),
          field(p17), field(p18), field(p19), field(p20), field(p21), field(p22), field(p23), field(p24),
          field(p25), field(p26), field(p27), field(p28), field(p29), field(p30), field(p31), field(p32)])

for elem in arr:
    print hex(int(elem))



DEVICE_FUNC void _basic_serial_radix2_FFT(embedded_field* arr, size_t log_arr_len, size_t omega_idx_coeff, bool is_inverse_FFT)
{
	size_t tid = threadIdx.x;
	size_t arr_len = 1 << log_arr_len;

	for(size_t i = tid; i < arr_len; i+= blockDim.x)
	{	
		size_t rk = __brev(i) >> (32 - log_arr_len);
		if (i < rk)
		{	
			embedded_field temp = arr[i];
			arr[i] = arr[rk];
			arr[rk] = temp;
		}
	}

	__syncthreads();
	
    for (size_t step = 0; step < log_arr_len; ++step)
    {
        uint32_t i = tid;
		uint32_t k = (1 << step);
		uint32_t l = 2 * k;
		while (i < arr_len / 2)
		{
			uint32_t first_index = l * (i / k) + (i % k);
			uint32_t second_index = first_index + k;

			uint32_t omega_idx = (1 << (log_arr_len - step - 1)) * (i % k); 
			embedded_field omega = get_root_of_unity(omega_idx, omega_idx_coeff, is_inverse_FFT);

			field_pair ops = fft_buttefly(arr[first_index], arr[second_index], omega);

			arr[first_index] = ops.a;
			arr[second_index] = ops.b;

			i += blockDim.x;
		}
		
		__syncthreads();
	}
}

__global__ void _basic_parallel_radix2_FFT(const embedded_field* input_arr, embedded_field* output_arr, size_t log_arr_len, 
	size_t log_num_subblocks, bool is_inverse_FFT)
{
    extern __shared__ embedded_field temp_arr[];

	assert( log_arr_len <= ROOTS_OF_UNTY_ARR_LEN && "the size of array is too large for FFT");

	size_t omega_coeff = 1 << (ROOTS_OF_UNTY_ARR_LEN - log_arr_len);
	size_t L = 1 << (log_arr_len - log_num_subblocks);
	size_t NUM_SUBBLOCKS = 1 << log_num_subblocks;

	embedded_field omega_step = get_root_of_unity(blockIdx.x * L, omega_coeff, is_inverse_FFT);
        
    for (size_t i = threadIdx.x; i < L; i+= blockDim.x)
    {
        embedded_field omega_init = get_root_of_unity(blockIdx.x * i, omega_coeff, is_inverse_FFT);
		temp_arr[i] = embedded_field::zero();
		for (size_t s = 0; s < NUM_SUBBLOCKS; ++s)
        {
            size_t idx = i + s * L;
            temp_arr[i] += input_arr[idx] * omega_init;
            omega_init *= omega_step;
        }
	}

	__syncthreads();

	_basic_serial_radix2_FFT(temp_arr, log_arr_len - log_num_subblocks, NUM_SUBBLOCKS * omega_coeff, is_inverse_FFT);

	for (size_t i = threadIdx.x; i < L; i+= blockDim.x)
		output_arr[i * NUM_SUBBLOCKS + blockIdx.x] = temp_arr[i];
}

__global__ void _radix2_one_block_FFT(const embedded_field* input_arr, embedded_field* output_arr, size_t log_arr_len, bool is_inverse_FFT)
{
	extern __shared__ embedded_field temp_arr[];
	size_t arr_len = 1 << log_arr_len;
	size_t omega_coeff = 1 << (ROOTS_OF_UNTY_ARR_LEN - log_arr_len);

	
	for (size_t i = threadIdx.x; i < arr_len; i+= blockDim.x)
	{
		temp_arr[i] = input_arr[i];
	}

	_basic_serial_radix2_FFT(temp_arr, log_arr_len, omega_coeff, is_inverse_FFT);

	for (size_t i = threadIdx.x; i < arr_len; i+= blockDim.x)
		output_arr[i] = temp_arr[i];
}

geometry find_geometry_for_advanced_FFT(uint arr_len)
{
	geometry res;

	size_t DEFAULT_FFT_GRID_SIZE = 16;

	return {DEFAULT_FFT_GRID_SIZE, arr_len / (2 * DEFAULT_FFT_GRID_SIZE)};

	//return geometry{4, 4};
	
	//TODO: apply some better heuristics!
	size_t DEFAULT_FFT_BLOCK_SIZE = 128;

	if (arr_len  <  2 * DEFAULT_FFT_BLOCK_SIZE)
	{
		res.gridSize = 1;
		res.blockSize = max(arr_len / 2, 1);
	}
	else
	{
		res.gridSize = arr_len / (2 * DEFAULT_FFT_BLOCK_SIZE);
		res.blockSize = DEFAULT_FFT_BLOCK_SIZE;
	}
	
	std::cout << "grid_size: " << res.gridSize << ", block size: " << res.blockSize << std::endl;
	return res;
}

void advanced_fft_driver(embedded_field* input_arr, embedded_field* output_arr, uint32_t arr_len, bool is_inverse_FFT = false)
{
	//first check that arr_len is a power of 2

	uint log_arr_len = BITS_PER_LIMB - __builtin_clz(arr_len) - 1;
    assert(arr_len = (1 << log_arr_len));

	

	//find optimal geometry

	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

	uint32_t shared_mem_per_block = prop.sharedMemPerBlock;
    std::cout << "SIZE OF SHARED ARR: " << shared_mem_per_block / sizeof(embedded_field) << std::endl;


	geometry kernel_geometry = find_geometry_for_advanced_FFT(arr_len);

	if (kernel_geometry.gridSize == 1)
	{
		std::cout << "We are now here!" << std::endl;
		
		_radix2_one_block_FFT<<<1, kernel_geometry.blockSize, kernel_geometry.blockSize * 2 * sizeof(embedded_field)>>>(input_arr, output_arr, 
			log_arr_len, is_inverse_FFT);
		cudaDeviceSynchronize();

		return;
	}

	uint log_num_subblocks = BITS_PER_LIMB - __builtin_clz(kernel_geometry.gridSize) - 1;
	std::cout << "log arr len: " << log_arr_len << ", log_num_subblocks: " << log_num_subblocks << std::endl;
	std::cout << "Initialization phase size: " <<  (1 << (log_arr_len - log_num_subblocks)) << std::endl;

	_basic_parallel_radix2_FFT<<<kernel_geometry.gridSize, kernel_geometry.blockSize, 
		kernel_geometry.blockSize * 2 * sizeof(embedded_field)>>>(input_arr, output_arr, 
		log_arr_len, log_num_subblocks, is_inverse_FFT);
	cudaDeviceSynchronize();
}
    
