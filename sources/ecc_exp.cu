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

DEVICE_FUNC static inline uint32_t convert_to_non_adjacent_form(const uint256_g& power, wnaf_auxiliary_data* form)
{
    uint32_t elem_count = 0;
    uint256_g d = power;
    uint8_t current_gap = 0;

    while (!is_zero(d))
    {
        if (is_even(d))
        {
            int8_t val = d.n[0] & EXP2(WINDOW_SIZE);
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
            d = SHIFT_RIGHT(d, WINDOW_SIZE);                     
        }
        else
        {
            current_gap++;
            d = SHIFT_RIGHT(d, 1);   
        }
    }

    return elem_count;
}

//TODO: may be better convert to affine?

DEVICE_FUNC ec_point ECC_wNAF_exp(const ec_point& pt, const uint256_g& power)
{
    if (is_zero(power))
        return point_at_infty();

    //precompute small powers

    ec_point precomputed[PRECOMPUTED_ARRAY_LEN];
    wnaf_auxiliary_data wnaf_arr[MAX_WNAF_DATA_ARRAY_LEN];
    
    ec_point pt_doubled = ECC_DOUBLE_PROJ(pt);
    precomputed[0] = pt;

    for (uint32_t i = 1; i < PRECOMPUTED_ARRAY_LEN; i++)
    {
        precomputed[i] = ECC_ADD_PROJ(precomputed[i-1], pt_doubled);
    }

    //convert degree to wNAF-form
    auto count = convert_to_non_adjacent_form(power, wnaf_arr);
    ec_point Q = point_at_infty();
    
    for (int j = count - 1; j >=0 ; j--)
    {
        auto& wnaf_data = wnaf_arr[j];
        if (wnaf_data.value >= 0)
            Q = ECC_ADD_PROJ(Q, precomputed[(wnaf_data.value - 1)/ 2]);
        else
            Q = ECC_SUB_PROJ(Q, precomputed[(-wnaf_data.value - 1)/ 2]);
        
        for(uint8_t k = 0; k < wnaf_data.gap; k++)
            Q = ECC_DOUBLE_PROJ(Q);
    }
   
   return Q;   
}


