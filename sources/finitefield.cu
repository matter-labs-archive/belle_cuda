#include "cuda_structs.h"

//This module comtains functions required for finite field arithmetic
//----------------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC uint256_g FIELD_ADD_INV(const uint256_g& elem)
{
    if (!is_zero(elem))
		return SUB(BASE_FIELD_P, elem);
	else
		return elem;
}

DEVICE_FUNC uint256_g FIELD_ADD(const uint256_g& a, const uint256_g& b )
{
    uint256_g w = ADD(a, b);
	if (CMP(w, BASE_FIELD_P) >= 0)
		return SUB(w, BASE_FIELD_P);
	return w;
}

DEVICE_FUNC uint256_g FIELD_SUB(const uint256_g& a, const uint256_g& b)
{
    if (CMP(a, b) > 0)
		return SUB(a, b);
	else
	{
		uint256_g t = ADD(a, BASE_FIELD_P);
		return SUB(t, b);
	}
}

//We are using https://www.researchgate.net/publication/3387259_Improved_Montgomery_modular_inverse_algorithm (algorithm 5)
//the description of The Almost Montgomery Inverse (so-called phase 1) is taken from 
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.75.8377&rep=rep1&type=pdf

struct stage_one_data
{
	uint256_g almost_mont_inverse;
	uint32_t k;
};

#include <stdio.h>

__device__ void print2_uint256(const uint256_g& val)
{
    printf("%x %x %x %x %x %x %x %x\n", val.n[7], val.n[6], val.n[5], val.n[4], val.n[3], val.n[2], val.n[1], val.n[0]);
}

static DEVICE_FUNC inline stage_one_data stage_one_mul_inv(const uint256_g& elem)
{
	uint256_g U = BASE_FIELD_P;
	uint256_g V = elem;
	uint256_g R = uint256_g{0, 0, 0, 0, 0, 0, 0, 0};
	uint256_g S = uint256_g{1, 0, 0, 0, 0, 0, 0, 0};

	uint32_t k = 0;

	while (!is_zero(V))
	{
		if (is_even(U))
		{
			U = SHIFT_RIGHT(U, 1);
			S = SHIFT_LEFT(S, 1);
		}
		else if (is_even(V))
		{
			V = SHIFT_RIGHT(V, 1);
			R = SHIFT_LEFT(R, 1);
		}
		else if (CMP(U, V) > 0)
		{
			U = SHIFT_RIGHT(SUB(U, V), 1);
			R = ADD(R, S);
			S = SHIFT_LEFT(S, 1);
		}
		else
		{
			V = SHIFT_RIGHT(SUB(V, U), 1);
			S = ADD(R, S);
			R = SHIFT_LEFT(R, 1);
		}

		k++;
	}

	if (CMP(R, BASE_FIELD_P) >= 0)
		R = SUB(R, BASE_FIELD_P);

	R = SUB(BASE_FIELD_P, R);

	return stage_one_data{R, k};
}

DEVICE_FUNC uint256_g FIELD_MUL_INV(const uint256_g& elem)
{
	auto data = stage_one_mul_inv(elem);
	if (data.k == R_LOG)
	{
		return MONT_MUL(data.almost_mont_inverse, BASE_FIELD_R2);
	}
	else
	{
		uint32_t n = 2 * R_LOG - data.k;
		auto res = uint256_g{0, 0, 0, 0, 0, 0, 0, 0};
		
		if (n < R_LOG)
		{		
			set_bit(res, n);
			res = MONT_MUL(data.almost_mont_inverse, res);
		}
		else if (n == R_LOG + 1)
		{
			res = MONT_MUL(data.almost_mont_inverse,  BASE_FIELD_R2);
		}
		else
		{
			//here n == R_LOG_2 + 2
			res = MONT_MUL(data.almost_mont_inverse,  BASE_FIELD_R4);
		}

		return MONT_MUL(res, BASE_FIELD_R_SQUARED);
	}
}

//batch inversion - simulaneously (in place) invert all-non zero elements in the array.
//NB: we assume that all elements in the array are non-zero


DEVICE_FUNC void BATCH_FIELD_MUL_INV(uint256_g* vec, size_t vec_size)
{
	
}