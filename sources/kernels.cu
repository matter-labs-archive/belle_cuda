#include "cuda_structs.h"
#include "cuda_macros.h"

#include <iostream>

GENERAL_TEST_2_ARGS_1_TYPE(add_uint256_naive, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(add_uint256_asm, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(sub_uint256_naive, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(sub_uint256_asm, uint256_g)

GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_naive, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm_with_allocation, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm_longregs, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_Karatsuba, uint256_g, uint512_g)
GENERAL_TEST_2_ARGS_2_TYPES(mul_uint256_to_512_asm_with_shuffle, uint256_g, uint512_g)


GENERAL_TEST_1_ARG_2_TYPES(square_uint256_to_512_naive, uint256_g, uint512_g)
GENERAL_TEST_1_ARG_2_TYPES(square_uint256_to_512_asm, uint256_g, uint512_g)

GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_naive_SOS, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_naive_CIOS, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_asm_SOS, uint256_g)
GENERAL_TEST_2_ARGS_1_TYPE(mont_mul_256_asm_CIOS, uint256_g)

GENERAL_TEST_2_ARGS_1_TYPE(ECC_ADD_PROJ, ec_point);
GENERAL_TEST_2_ARGS_1_TYPE(ECC_SUB_PROJ, ec_point);
GENERAL_TEST_1_ARG_1_TYPE(ECC_DOUBLE_PROJ, ec_point);
GENERAL_TEST_1_ARG_2_TYPES(IS_ON_CURVE_PROJ, ec_point, bool);

GENERAL_TEST_2_ARGS_1_TYPE(ECC_ADD_JAC, ec_point);
GENERAL_TEST_2_ARGS_1_TYPE(ECC_SUB_JAC, ec_point);
GENERAL_TEST_1_ARG_1_TYPE(ECC_DOUBLE_JAC, ec_point);
GENERAL_TEST_1_ARG_2_TYPES(IS_ON_CURVE_JAC, ec_point, bool);

GENERAL_TEST_2_ARGS_3_TYPES(ECC_double_and_add_exp_PROJ, ec_point, uint256_g, ec_point);
GENERAL_TEST_2_ARGS_3_TYPES(ECC_ternary_expansion_exp_PROJ, ec_point, uint256_g, ec_point);
GENERAL_TEST_2_ARGS_3_TYPES(ECC_double_and_add_exp_JAC, ec_point, uint256_g, ec_point);
GENERAL_TEST_2_ARGS_3_TYPES(ECC_ternary_expansion_exp_JAC, ec_point, uint256_g, ec_point);