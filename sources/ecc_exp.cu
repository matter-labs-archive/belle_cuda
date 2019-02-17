#include "cuda_structs.h"


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

//algorthm with ternary expansion (TODO: have a look at Pomerance prime numbers book)
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
                y = 0;\
                z = 1;\
            }\
            else if (!x)\
            {\
                ec_point temp = (z ? INV(R) : R);\
                Q = ECC_ADD##SUFFIX(Q, temp);\
            }\
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


//Ddecreaing version of double and add algorithm in order to be able to use mixed addition
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

#define DOUBLE_AND_ADD_AFFINE_EXP(SUFFIX) \
DEVICE_FUNC ec_point ECC_double_and_add_affine_exp##SUFFIX(const affine_point& pt, const uint256_g& power)\
{\
	ec_point Q = point_at_infty();\
\
	for (int i = N_BITLEN - 1; i >= 0; i--)\
	{\
		Q = ECC_DOUBLE##SUFFIX(Q);\
        bool flag = get_bit(power, i);\
		if (flag)\
        {\
            Q = ECC_ADD_MIXED##SUFFIX(Q, pt);\
        }\
	}\
	return Q;\
}

DOUBLE_AND_ADD_AFFINE_EXP(_PROJ)
DOUBLE_AND_ADD_AFFINE_EXP(_JAC)


//Wnaf method
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

//TODO: allign this struct in order to require very little amount of space
struct wnaf_auxiliary_data
{
    //NB: value is signed!
    int8_t value;
    uint8_t gap;
};

//NB: WINDOW_SIZE should be >= 4
#define WINDOW_SIZE 4
#define EXP2(w) (1 << w)
#define EXP2_MINUS_1(w) (1 << (w - 1))

static constexpr uint32_t PRECOMPUTED_ARRAY_LEN = (1 << (WINDOW_SIZE - 2)) - 2;
static constexpr uint32_t MAX_WNAF_DATA_ARRAY_LEN = (N_BITLEN / WINDOW_SIZE ) + 1;

//NB: we assume that power is nonzero here
//returns the number of wnaf_auxiliary_data in form array
//we also assume that bit-len of w is less than word-size

using clock_value_t = long long;

__device__ void sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

#include <stdio.h>

__device__ void print_uint256(const uint256_g& val)
{
    printf("%x %x %x %x %x %x %x %x\n", val.n[7], val.n[6], val.n[5], val.n[4], val.n[3], val.n[2], val.n[1], val.n[0]);
}

DEVICE_FUNC static inline uint32_t convert_to_non_adjacent_form(const uint256_g& power, wnaf_auxiliary_data* form)
{
    uint32_t elem_count = 0;
    uint256_g d = power;
    uint8_t current_gap = 0;

    while (!is_zero(d))
    {
        uint32_t pos = __ffs(d.n[0]);
        uint32_t shift;
        if (pos == 1)
        {
            int8_t val = d.n[0] & (EXP2(WINDOW_SIZE) - 1);
            if (val >= EXP2_MINUS_1(WINDOW_SIZE))
            {
                val -= EXP2(WINDOW_SIZE);
                ADD_UINT(d, -val);
            }
            else
            {
                SUB_UINT(d, val);
            }
          
            form[elem_count++] = {val, current_gap};
            current_gap = WINDOW_SIZE;
            shift = WINDOW_SIZE;  
        }
        else
        {
            shift = min(pos - 1, 32);
            current_gap += shift;
            shift = 32;
        }

        d = SHIFT_RIGHT(d, shift);    
    }

    return elem_count;
}

#define ECC_WNAF_EXP(SUFFIX) \
DEVICE_FUNC ec_point ECC_wNAF_exp##SUFFIX(const ec_point& pt, const uint256_g& power)\
{\
    if (is_zero(power))\
        return point_at_infty();\
\
    ec_point precomputed[PRECOMPUTED_ARRAY_LEN];\
    wnaf_auxiliary_data wnaf_arr[MAX_WNAF_DATA_ARRAY_LEN];\
\
    ec_point pt_doubled = ECC_DOUBLE##SUFFIX(pt);\
    precomputed[0] = pt;\
\
    for (uint32_t i = 1; i < PRECOMPUTED_ARRAY_LEN; i++)\
    {\
        precomputed[i] = ECC_ADD##SUFFIX(precomputed[i-1], pt_doubled);\
    }\
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
}

ECC_WNAF_EXP(_PROJ)
ECC_WNAF_EXP(_JAC)


