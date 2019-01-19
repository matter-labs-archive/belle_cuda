#include "cuda_structs.h"

//miscellaneous helpful staff
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------


DEVICE_FUNC inline bool get_bit(const uint256_g& x, size_t index)
{
	auto num = x.n[index / 32];
	auto pos = index % 32;
	return CHECK_BIT(num, pos);
}

//classical double and add algorithm:
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

//NB: may be we can achieve additional sppedup by using special logic for +=

#define DOUBLE_AND_ADD_EXP(SUFFIX) \
DEVICE_FUNC ec_point ECC_double_and_add_exp##SUFFIX(const ec_point& pt, const uint256_g& power)\
{\
    ec_point R = pt;\
	ec_point Q = point_at_infty();\
\
	for (size_t i = 0; i < N_BITLEN; i++)\
	{\
		bool flag = get_bit(power, i);\
		if (flag)\
        {\
            Q = ECC_ADD##SUFFIX(Q, R);\
        }\
        R = ECC_DOUBLE##SUFFIX(R);\
	}\
	return Q;\
}

DOUBLE_AND_ADD_EXP(_PROJ)
DOUBLE_AND_ADD_EXP(_JAC)

//algorthm with ternary expansion
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

#define TERNARY_EXPANSION_EXP(SUFFIX) \
DEVICE_FUNC ec_point ECC_ternary_expansion_exp##SUFFIX(const ec_point& pt, const uint256_g& power)\
{\
    ec_point R = pt;\
	ec_point Q = point_at_infty();\
\
    bool x = false;\
    bool y = get_bit(power, 0);\
    bool z;\
\
	for (size_t i = 0; i < N_BITLEN; i++)\
	{\
	    z = get_bit(power, i + 1);\
        if (y)\
        {\
            if (x && !z)\
            {\
                x = 0;\
                y = 1;\
                R = ECC_DOUBLE##SUFFIX(R);\
                continue;\
            }\
            if (!x && !z)\
                Q = ECC_ADD##SUFFIX(Q, R);\
            if (!x && z)\
                Q = ECC_SUB##SUFFIX(Q, R);\
        }\
\
        x = y;\
        y = z;\
        R = ECC_DOUBLE##SUFFIX(R);\
	}\
\
    if (y)\
      Q = ECC_ADD##SUFFIX(Q, R);\
	return Q;\
}

TERNARY_EXPANSION_EXP(_PROJ)
TERNARY_EXPANSION_EXP(_JAC)

