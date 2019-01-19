#include "cuda_structs.h"

DEVICE_FUNC uint512_g mul_uint256_to_512_naive(const uint256_g& u, const uint256_g& v)
{
    uint512_g w;
    #pragma unroll
	for (uint32_t j = 0; j < N_DOUBLED; j++)
        w.n[j] = 0;
		
    #pragma unroll
	for (uint32_t j = 0; j < N; j++)
	{
        uint32_t k = 0;

        #pragma unroll
        for (uint32_t i = 0; i < N; i++)
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

DEVICE_FUNC uint512_g mul_uint256_to_512_asm(const uint256_g& lhs, const uint256_g& rhs)
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
            "addc.cc.u32   %9, %9, 0;\n\t"
            "addc.cc.u32   %10, %10, 0;\n\t"
            "addc.cc.u32   %11, %11, 0;\n\t"
            "addc.cc.u32   %12, %12, 0;\n\t"
            "addc.cc.u32   %13, %13, 0;\n\t"
            "addc.cc.u32   %14, %14, 0;\n\t"
            "addc.cc.u32   %15, %15, 0;\n\t"
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

DEVICE_FUNC uint512_g mul_uint256_to_512_asm_with_allocation(const uint256_g& lhs, const uint256_g& rhs)
{
    uint512_g w;
		
    asm (   ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
            ".reg .u32 r8, r9, r10, r11, r12, r13, r14, r15;\n\t"
            ".reg .u32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
            ".reg .u32 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
            "mov.b64         {a0,a1}, %8;\n\t"
            "mov.b64         {a2,a3}, %9;\n\t"
            "mov.b64         {a4,a5}, %10;\n\t"
            "mov.b64         {a6,a7}, %11;\n\t"
            "mov.b64         {b0,b1}, %12;\n\t"
            "mov.b64         {b2,b3}, %13;\n\t"
            "mov.b64         {b4,b5}, %14;\n\t"
            "mov.b64         {b6,b7}, %15;\n\t"   
            "mul.lo.u32   r0, a0, b0;\n\t"
            "mul.hi.u32   r1, a0, b0;\n\t"
            "mad.lo.cc.u32   r1, a1, b0, r1;\n\t"
            "madc.hi.u32   r2, a1, b0, 0;\n\t"
            "mad.lo.cc.u32   r1, a0, b1, r1;\n\t"
            "madc.hi.cc.u32   r2, a0, b1, r2;\n\t"
            "madc.hi.u32   r3, a2, b0, 0;\n\t"
            "mad.lo.cc.u32   r2, a2, b0, r2;\n\t"
            "madc.hi.cc.u32   r3, a1, b1, r3;\n\t"
            "madc.hi.u32   r4, a3, b0, 0;\n\t"
            "mad.lo.cc.u32   r2, a1, b1, r2;\n\t"
            "madc.hi.cc.u32   r3, a0, b2, r3;\n\t"
            "madc.hi.cc.u32   r4, a2, b1, r4;\n\t"
            "madc.hi.u32   r5, a4, b0, 0;\n\t"
            "mad.lo.cc.u32   r2, a0, b2, r2;\n\t"
            "madc.lo.cc.u32   r3, a3, b0, r3;\n\t"
            "madc.hi.cc.u32   r4, a1, b2, r4;\n\t"
            "madc.hi.cc.u32   r5, a3, b1, r5;\n\t"
            "madc.hi.u32   r6, a5, b0, 0;\n\t"
            "mad.lo.cc.u32   r3, a2, b1, r3;\n\t"
            "madc.hi.cc.u32   r4, a0, b3, r4;\n\t"
            "madc.hi.cc.u32   r5, a2, b2, r5;\n\t"
            "madc.hi.cc.u32   r6, a4, b1, r6;\n\t"
            "madc.hi.u32   r7, a6, b0, 0;\n\t"
            "mad.lo.cc.u32   r3, a1, b2, r3;\n\t"
            "madc.lo.cc.u32   r4, a4, b0, r4;\n\t"
            "madc.hi.cc.u32   r5, a1, b3, r5;\n\t"
            "madc.hi.cc.u32   r6, a3, b2, r6;\n\t"
            "madc.hi.cc.u32   r7, a5, b1, r7;\n\t"
            "madc.hi.u32   r8, a7, b0, 0;\n\t"
            "mad.lo.cc.u32   r3, a0, b3, r3;\n\t"
            "madc.lo.cc.u32   r4, a3, b1, r4;\n\t"
            "madc.hi.cc.u32   r5, a0, b4, r5;\n\t"
            "madc.hi.cc.u32   r6, a2, b3, r6;\n\t"
            "madc.hi.cc.u32   r7, a4, b2, r7;\n\t"
            "madc.hi.cc.u32   r8, a6, b1, r8;\n\t"
            "madc.hi.u32   r9, a7, b1, 0;\n\t"
            "mad.lo.cc.u32   r4, a2, b2, r4;\n\t"
            "madc.lo.cc.u32   r5, a5, b0, r5;\n\t"
            "madc.hi.cc.u32   r6, a1, b4, r6;\n\t"
            "madc.hi.cc.u32   r7, a3, b3, r7;\n\t"
            "madc.hi.cc.u32   r8, a5, b2, r8;\n\t"
            "madc.hi.cc.u32   r9, a6, b2, r9;\n\t"
            "madc.hi.u32   r10, a7, b2, 0;\n\t"
            "mad.lo.cc.u32   r4, a1, b3, r4;\n\t"
            "madc.lo.cc.u32   r5, a4, b1, r5;\n\t"
            "madc.hi.cc.u32   r6, a0, b5, r6;\n\t"
            "madc.hi.cc.u32   r7, a2, b4, r7;\n\t"
            "madc.hi.cc.u32   r8, a4, b3, r8;\n\t"
            "madc.hi.cc.u32   r9, a5, b3, r9;\n\t"
            "madc.hi.cc.u32   r10, a6, b3, r10;\n\t"
            "madc.hi.u32   r11, a7, b3, 0;\n\t"
            "mad.lo.cc.u32   r4, a0, b4, r4;\n\t"
            "madc.lo.cc.u32   r5, a3, b2, r5;\n\t"
            "madc.lo.cc.u32   r6, a6, b0, r6;\n\t"
            "madc.hi.cc.u32   r7, a1, b5, r7;\n\t"
            "madc.hi.cc.u32   r8, a3, b4, r8;\n\t"
            "madc.hi.cc.u32   r9, a4, b4, r9;\n\t"
            "madc.hi.cc.u32   r10, a5, b4, r10;\n\t"
            "madc.hi.cc.u32   r11, a6, b4, r11;\n\t"
            "madc.hi.u32   r12, a7, b4, 0;\n\t"
            "mad.lo.cc.u32   r5, a2, b3, r5;\n\t"
            "madc.lo.cc.u32   r6, a5, b1, r6;\n\t"
            "madc.hi.cc.u32   r7, a0, b6, r7;\n\t"
            "madc.hi.cc.u32   r8, a2, b5, r8;\n\t"
            "madc.hi.cc.u32   r9, a3, b5, r9;\n\t"
            "madc.hi.cc.u32   r10, a4, b5, r10;\n\t"
            "madc.hi.cc.u32   r11, a5, b5, r11;\n\t"
            "madc.hi.cc.u32   r12, a6, b5, r12;\n\t"
            "madc.hi.u32   r13, a7, b5, 0;\n\t"
            "mad.lo.cc.u32   r5, a1, b4, r5;\n\t"
            "madc.lo.cc.u32   r6, a4, b2, r6;\n\t"
            "madc.lo.cc.u32   r7, a7, b0, r7;\n\t"
            "madc.hi.cc.u32   r8, a1, b6, r8;\n\t"
            "madc.hi.cc.u32   r9, a2, b6, r9;\n\t"
            "madc.hi.cc.u32   r10, a3, b6, r10;\n\t"
            "madc.hi.cc.u32   r11, a4, b6, r11;\n\t"
            "madc.hi.cc.u32   r12, a5, b6, r12;\n\t"
            "madc.hi.cc.u32   r13, a6, b6, r13;\n\t"
            "madc.hi.u32   r14, a7, b6, 0;\n\t"
            "mad.lo.cc.u32   r5, a0, b5, r5;\n\t"
            "madc.lo.cc.u32   r6, a3, b3, r6;\n\t"
            "madc.lo.cc.u32   r7, a6, b1, r7;\n\t"
            "madc.hi.cc.u32   r8, a0, b7, r8;\n\t"
            "madc.hi.cc.u32   r9, a1, b7, r9;\n\t"
            "madc.hi.cc.u32   r10, a2, b7, r10;\n\t"
            "madc.hi.cc.u32   r11, a3, b7, r11;\n\t"
            "madc.hi.cc.u32   r12, a4, b7, r12;\n\t"
            "madc.hi.cc.u32   r13, a5, b7, r13;\n\t"
            "madc.hi.cc.u32   r14, a6, b7, r14;\n\t"
            "madc.hi.u32   r15, a7, b7, 0;\n\t"
            "mad.lo.cc.u32   r6, a2, b4, r6;\n\t"
            "madc.lo.cc.u32   r7, a5, b2, r7;\n\t"
            "madc.lo.cc.u32   r8, a7, b1, r8;\n\t"
            "madc.lo.cc.u32   r9, a7, b2, r9;\n\t"
            "madc.lo.cc.u32   r10, a7, b3, r10;\n\t"
            "madc.lo.cc.u32   r11, a7, b4, r11;\n\t"
            "madc.lo.cc.u32   r12, a7, b5, r12;\n\t"
            "madc.lo.cc.u32   r13, a7, b6, r13;\n\t"
            "madc.lo.cc.u32   r14, a7, b7, r14;\n\t"
            "addc.cc.u32   r15, r15, 0;\n\t"
            "mad.lo.cc.u32   r6, a1, b5, r6;\n\t"
            "madc.lo.cc.u32   r7, a4, b3, r7;\n\t"
            "madc.lo.cc.u32   r8, a6, b2, r8;\n\t"
            "madc.lo.cc.u32   r9, a6, b3, r9;\n\t"
            "madc.lo.cc.u32   r10, a6, b4, r10;\n\t"
            "madc.lo.cc.u32   r11, a6, b5, r11;\n\t"
            "madc.lo.cc.u32   r12, a6, b6, r12;\n\t"
            "madc.lo.cc.u32   r13, a6, b7, r13;\n\t"
            "addc.cc.u32   r14, r14, 0;\n\t"
            "addc.cc.u32   r15, r15, 0;\n\t"
            "mad.lo.cc.u32   r6, a0, b6, r6;\n\t"
            "madc.lo.cc.u32   r7, a3, b4, r7;\n\t"
            "madc.lo.cc.u32   r8, a5, b3, r8;\n\t"
            "madc.lo.cc.u32   r9, a5, b4, r9;\n\t"
            "madc.lo.cc.u32   r10, a5, b5, r10;\n\t"
            "madc.lo.cc.u32   r11, a5, b6, r11;\n\t"
            "madc.lo.cc.u32   r12, a5, b7, r12;\n\t"
            "addc.cc.u32   r13, r13, 0;\n\t"
            "addc.cc.u32   r14, r14, 0;\n\t"
            "addc.cc.u32   r15, r15, 0;\n\t"
            "mad.lo.cc.u32   r7, a2, b5, r7;\n\t"
            "madc.lo.cc.u32   r8, a4, b4, r8;\n\t"
            "madc.lo.cc.u32   r9, a4, b5, r9;\n\t"
            "madc.lo.cc.u32   r10, a4, b6, r10;\n\t"
            "madc.lo.cc.u32   r11, a4, b7, r11;\n\t"
            "addc.cc.u32   r12, r12, 0;\n\t"
            "addc.cc.u32   r13, r13, 0;\n\t"
            "addc.cc.u32   r14, r14, 0;\n\t"
            "addc.cc.u32   r15, r15, 0;\n\t"
            "mad.lo.cc.u32   r7, a1, b6, r7;\n\t"
            "madc.lo.cc.u32   r8, a3, b5, r8;\n\t"
            "madc.lo.cc.u32   r9, a3, b6, r9;\n\t"
            "madc.lo.cc.u32   r10, a3, b7, r10;\n\t"
            "addc.cc.u32   r11, r11, 0;\n\t"
            "addc.cc.u32   r12, r12, 0;\n\t"
            "addc.cc.u32   r13, r13, 0;\n\t"
            "addc.cc.u32   r14, r14, 0;\n\t"
            "addc.cc.u32   r15, r15, 0;\n\t"
            "mad.lo.cc.u32   r7, a0, b7, r7;\n\t"
            "madc.lo.cc.u32   r8, a2, b6, r8;\n\t"
            "madc.lo.cc.u32   r9, a2, b7, r9;\n\t"
            "addc.cc.u32   r10, r10, 0;\n\t"
            "addc.cc.u32   r11, r11, 0;\n\t"
            "addc.cc.u32   r12, r12, 0;\n\t"
            "addc.cc.u32   r13, r13, 0;\n\t"
            "addc.cc.u32   r14, r14, 0;\n\t"
            "addc.cc.u32   r15, r15, 0;\n\t"
            "mad.lo.cc.u32   r8, a1, b7, r8;\n\t"
            "addc.cc.u32   r9, r9, 0;\n\t"
            "addc.cc.u32   r10, r10, 0;\n\t"
            "addc.cc.u32   r11, r11, 0;\n\t"
            "addc.cc.u32   r12, r12, 0;\n\t"
            "addc.cc.u32   r13, r13, 0;\n\t"
            "addc.cc.u32   r14, r14, 0;\n\t"
            "addc.cc.u32   r15, r15, 0;\n\t"
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
            : "l"(lhs.nn[0]), "l"(lhs.nn[1]), "l"(lhs.nn[2]), "l"(lhs.nn[3]),
                "l"(rhs.nn[0]), "l"(rhs.nn[1]), "l"(rhs.nn[2]), "l"(rhs.nn[3]));

    return w;	
}

//the same logic as multiplication 128 x 128 -> 256
//but we consider limbs as 64 bit

//NB: https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/
//There are intersting considerations on 16 bit registers. May be we should use them?

DEVICE_FUNC uint512_g mul_uint256_to_512_asm_longregs(const uint256_g& lhs, const uint256_g& rhs)
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

//let u = u1 + u2 * 2^n
//let v = v1 + v2 * 2^n
//result = (u1 * v1) + 2^n * ((u1 + u2)(v1 + v2) - (u1 * v1) - (u2 * v2))  + 2^(2n) u2 * v2;
//hence we require more space and addition operations bu only 3 multiplications (instead of four)

DEVICE_FUNC uint512_g mul_uint256_to_512_Karatsuba(const uint256_g& u, const uint256_g& v)
{
    uint256_g x = MUL_SHORT(u.low, v.low);
    uint256_g y = MUL_SHORT(u.high, v.high);
    uint128_with_carry_g a = add_uint128_with_carry_asm(u.low, u.high);
    uint128_with_carry_g b = add_uint128_with_carry_asm(v.low, v.high);
    uint256_g c = MUL_SHORT(a.val, b.val);

    uint512_g res;

     asm (  ".reg .pred p;\n\t"
            ".reg .u32 r4, r5, r6, r7, r8, r9, r10, r11;\n\t"
            ".reg .u32 r12, r13, r14, r15;\n\t"
            "mov.b64  %0, {%8, %9};\n\t"
            "mov.b64  %1, {%10, %11};\n\t"  
            
            "add.cc.u32  r4, %12, %16;\n\t"
            "addc.cc.u32 r5, %13, %17;\n\t"
            "addc.cc.u32 r6, %14, %18;\n\t"
            "addc.cc.u32 r7, %15, %19;\n\t"

            "addc.cc.u32 r8, %20, %24;\n\t"
            "addc.cc.u32 r9, %21, %25;\n\t"
            "addc.cc.u32 r10, %22, %26;\n\t"
            "addc.cc.u32 r11, %23, %27;\n\t"
            "addc.u32  r12, 0, 0;\n\t"

            "sub.cc.u32  r4, r4, %8;\n\t"
            "subc.cc.u32 r5, r5, %9;\n\t"
            "subc.cc.u32 r6, r6, %10;\n\t"
            "subc.cc.u32 r7, r7, %11;\n\t"
            "subc.cc.u32 r8, r8, %12;\n\t"
            "subc.cc.u32 r9, r9, %13;\n\t"
            "subc.cc.u32 r10, r10, %14;\n\t"
            "subc.cc.u32 r11, r11, %15;\n\t"
            "subc.u32  r12, r12, 0;\n\t"

            "sub.cc.u32  r4, r4, %24;\n\t"
            "subc.cc.u32 r5, r5, %25;\n\t"
            "subc.cc.u32 r6, r6, %26;\n\t"
            "subc.cc.u32 r7, r7, %27;\n\t"
            "subc.cc.u32 r8, r8, %28;\n\t"
            "subc.cc.u32 r9, r9, %29;\n\t"
            "subc.cc.u32 r10, r10, %30;\n\t"
            "subc.cc.u32 r11, r11, %31;\n\t"
            "subc.u32  r12, r12, 0;\n\t"

            "setp.eq.u32 p, %41, 0;\n\t"
            "@p  bra $label1; \n\t" 
            "addc.cc.u32 r8, r8, %32;\n\t"
            "addc.cc.u32 r9, r9, %33;\n\t"
            "addc.cc.u32 r10, r10, %34;\n\t"
            "addc.cc.u32 r11, r11, %35;\n\t"
            "addc.u32 r12, r12, 0;\n\t"

            "$label1:\n\t"
            "setp.eq.u32 p, %36, 0;\n\t"
            "@p  bra $label2; \n\t" 
            "addc.cc.u32 r8, r8, %37;\n\t"
            "addc.cc.u32 r9, r9, %38;\n\t"
            "addc.cc.u32 r10, r10, %39;\n\t"
            "addc.cc.u32 r11, r11, %40;\n\t"
            "addc.u32 r12, r12, 0;\n\t"

            "$label2:\n\t"
            "add.cc.u32  r12, r12, %28;\n\t"
            "addc.cc.u32  r13, 0, %29;\n\t"
            "addc.cc.u32  r14, 0, %30;\n\t"
            "addc.u32  r15, 0, %31;\n\t"

            "mov.b64         %2, {r4,r5};\n\t"  
            "mov.b64         %3, {r6,r7};\n\t"
            "mov.b64         %4, {r8,r9};\n\t"  
            "mov.b64         %5, {r10,r11};\n\t"
            "mov.b64         %6, {r12,r13};\n\t"  
            "mov.b64         %7, {r14,r15};\n\t"
         	 	
            : "=l"(res.nn[0]), "=l"(res.nn[1]),"=l"(res.nn[2]), "=l"(res.nn[3]),
            "=l"(res.nn[4]), "=l"(res.nn[5]),"=l"(res.nn[6]), "=l"(res.nn[7])
            : "r"(x.n[0]), "r"(x.n[1]), "r"(x.n[2]), "r"(x.n[3]), "r"(x.n[4]), "r"(x.n[5]), "r"(x.n[6]), "r"(x.n[7]),
            "r"(c.n[0]), "r"(c.n[1]), "r"(c.n[2]), "r"(c.n[3]), "r"(c.n[4]), "r"(c.n[5]), "r"(c.n[6]), "r"(c.n[7]),
            "r"(y.n[0]), "r"(y.n[1]), "r"(y.n[2]), "r"(y.n[3]), "r"(y.n[4]), "r"(y.n[5]), "r"(y.n[6]), "r"(y.n[7]),
            "r"(a.val.n[0]), "r"(a.val.n[1]), "r"(a.val.n[2]), "r"(a.val.n[3]), "r"(a.carry),
            "r"(b.val.n[0]), "r"(b.val.n[1]), "r"(b.val.n[2]), "r"(b.val.n[3]), "r"(b.carry));
        	
    return res;	
}

