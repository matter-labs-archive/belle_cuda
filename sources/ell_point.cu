#include "cuda_structs.h"

//Arithmetic in projective coordinates (Jacobian coordinates should be faster and we are going to check it!)
//TODO: we may also use BN specific optimizations (for example use, that a = 0)
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC ec_point ECC_DOUBLE_PROJ(const ec_point& pt)
{
	if (is_zero(pt.y) || is_infinity(pt))
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
 		temp = MONT_MUL(temp, BASE_FIELD_R3);
 		temp2 = MONT_SQUARE(pt.z);
 		temp2 = MONT_MUL(temp2, CURVE_A_COEFF);
 		W = FIELD_ADD(temp, temp2);
#endif
 		S = MONT_MUL(pt.y, pt.z);
		temp = MONT_MUL(pt.x, pt.y);
 		B = MONT_MUL(temp, S);
		res.x = W;

 		temp = MONT_SQUARE(W);
 		temp2 = MONT_MUL(BASE_FIELD_R8, B);
 		H = FIELD_SUB(temp, temp2);

 		temp = MONT_MUL(BASE_FIELD_R2, H);
 		res.x = MONT_MUL(temp, S);
		
 		//NB: here result is also equal to one of the operands and hence may be reused!!!
 		//NB: this is in fact another possibility for optimization!
 		S2 = MONT_SQUARE(S);
 		temp = MONT_MUL(BASE_FIELD_R4, B);
 		temp = FIELD_SUB(temp, H);
 		temp = MONT_MUL(W, temp);
		
 		temp2 = MONT_SQUARE(pt.y);
 		temp2 = MONT_MUL(BASE_FIELD_R8, temp2);
 		temp2 = MONT_MUL(temp2, S2);
 		res.y = FIELD_SUB(temp, temp2);

 		temp = MONT_MUL(BASE_FIELD_R8, S);
 		res.z = MONT_MUL(temp, S2);

		return res;
	}
}

//for debug purposes only: check if point is indeed on curve
DEVICE_FUNC bool IS_ON_CURVE_PROJ(const ec_point& pt)
{
	//y^{2} * z = x^{3} + A *x * z^{2} + B * z^{3}
	uint256_g temp1, temp2, z2; 
	z2 = MONT_SQUARE(pt.z);
	temp1 = MONT_SQUARE(pt.x);
	temp1 = MONT_MUL(temp1, pt.x);
	temp2 = MONT_MUL(CURVE_A_COEFF, pt.x);
	temp2 = MONT_MUL(temp2, z2);
	temp1 = FIELD_ADD(temp1, temp2);
	temp2 = MONT_MUL(CURVE_B_COEFF, pt.z);
	temp2 = MONT_MUL(temp2, z2);
	temp1 = FIELD_ADD(temp1, temp2);
	temp2 = MONT_SQUARE(pt.y);
	temp2 = MONT_MUL(temp2, pt.z);

	return EQUAL(temp1, temp2);
}

DEVICE_FUNC bool EQUAL_PROJ(const ec_point& pt1, const ec_point& pt2)
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

DEVICE_FUNC ec_point ECC_ADD_PROJ(const ec_point& left, const ec_point& right)
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
			return  ECC_DOUBLE_PROJ(left);
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
	temp2 = MONT_MUL(BASE_FIELD_R2, Vsq);
	temp2 = MONT_MUL(temp2, V2);
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

DEVICE_FUNC ec_point ECC_SUB_PROJ(const ec_point& left, const ec_point& right)
{
	return ECC_ADD_PROJ(left, INV(right));
}

DEVICE_FUNC ec_point ECC_ADD_MIXED_PROJ(const ec_point& left, const affine_point& right)
{
	if (is_infinity(left))
		return ec_point{right.x, right.y, BASE_FIELD_R};

	uint256_g U1, V1;
	U1 = MONT_MUL(left.z, right.y);
	V1 = MONT_MUL(left.z, right.x);

	ec_point res;

	if (EQUAL(V1, left.x))
	{
		if (!EQUAL(U1, left.y))
			return point_at_infty();
		else
			return  ECC_DOUBLE_PROJ(left);
	}

	uint256_g U = FIELD_SUB(U1, left.y);
	uint256_g V = FIELD_SUB(V1, left.x);
	uint256_g Vsq = MONT_SQUARE(V);
	uint256_g Vcube = MONT_MUL(Vsq, V);

	uint256_g temp1, temp2;
	temp1 = MONT_SQUARE(U);
	temp1 = MONT_MUL(temp1, left.z);
	temp1 = FIELD_SUB(temp1, Vcube);
	temp2 = MONT_MUL(BASE_FIELD_R2, Vsq);
	temp2 = MONT_MUL(temp2, left.x);
	uint256_g A = FIELD_SUB(temp1, temp2);
	res.x = MONT_MUL(V, A);

	temp1 = MONT_MUL(Vsq, left.x);
	temp1 = FIELD_SUB(temp1, A);
	temp1 = MONT_MUL(U, temp1);
	temp2 = MONT_MUL(Vcube, left.y);
	res.y = FIELD_SUB(temp1, temp2);

	res.z = MONT_MUL(Vcube, left.z);
	return res;
}

// Arithmetic in Jacobian coordinates (Jacobian coordinates should be faster and we are going to check it!)
// TODO: we may also use BN specific optimizations (for example use, that a = 0)
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------------------------------------

//TODO: An alternative repeated doubling routine with costs (4m)M + (4m+2)S for any value a can be derived from the Modified Jacobian doubling routine. 
// For small values a (say 0 or -3) the costs reduce to (4m-1)M + (4m+2)S, competing nicely with the algorithm showed above.


DEVICE_FUNC ec_point ECC_DOUBLE_JAC(const ec_point& pt)
{
	if (is_zero(pt.y) || is_infinity(pt))
		return point_at_infty();
	else
	{
		uint256_g temp1, temp2;
		temp1 = MONT_MUL(BASE_FIELD_R4, pt.x);
		uint256_g Ysq = MONT_SQUARE(pt.y);
		uint256_g S = MONT_MUL(temp1, Ysq);

//TODO: here we may also use BN-SPECIFIC optimizations, cause A = 0

		temp1 = MONT_SQUARE(pt.x);
		temp1 = MONT_MUL(BASE_FIELD_R3, temp1);
		temp2 = MONT_SQUARE(pt.z);
		temp2 = MONT_SQUARE(temp2);
		temp2 = MONT_MUL(temp2, CURVE_A_COEFF);
		uint256_g M = FIELD_ADD(temp1, temp2);

		temp1 = MONT_SQUARE(M);
		temp2 = MONT_MUL(BASE_FIELD_R2, S);
		uint256_g res_x = FIELD_SUB(temp1, temp2);
		
		temp1 = FIELD_SUB(S, res_x);
		temp1 = MONT_MUL(M, temp1);
		temp2 = MONT_SQUARE(Ysq);
		temp2 = MONT_MUL(BASE_FIELD_R8, temp2);
		uint256_g res_y = FIELD_SUB(temp1, temp2);

		temp1 = MONT_MUL(BASE_FIELD_R2, pt.y);
		uint256_g res_z = MONT_MUL(temp1, pt.z);

		return ec_point{res_x, res_y, res_z};
	}
}

DEVICE_FUNC bool IS_ON_CURVE_JAC(const ec_point& pt)
{
	//y^4 = x^3 + a  x z^4 +b z^6
	uint256_g temp1 = MONT_SQUARE(pt.y);
	uint256_g lefthandside = MONT_SQUARE(temp1);

	uint256_g Zsq = MONT_SQUARE(pt.z);
	uint256_g Z4 = MONT_SQUARE(Zsq);

	temp1 = MONT_SQUARE(pt.x);
	uint256_g righthandside = MONT_MUL(temp1, pt.x);
	temp1 = MONT_MUL(CURVE_A_COEFF, pt.x);
	temp1 = MONT_MUL(temp1, Z4);
	righthandside = FIELD_ADD(righthandside, temp1);
	temp1 = MONT_MUL(CURVE_B_COEFF, Zsq);
	temp1 = MONT_MUL(temp1, Z4);
	righthandside = FIELD_ADD(righthandside, temp1);

	return EQUAL(lefthandside, righthandside);
}

DEVICE_FUNC bool EQUAL_JAC(const ec_point& pt1, const ec_point& pt2)
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

DEVICE_FUNC ec_point ECC_ADD_JAC(const ec_point& left, const ec_point& right)
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
			return  ECC_DOUBLE_JAC(left);
	}

	uint256_g H = FIELD_SUB(U2, U1);
	uint256_g R = FIELD_SUB(S2, S1);
	uint256_g Hsq = MONT_SQUARE(H);
	uint256_g Hcube = MONT_MUL(Hsq, H);
	uint256_g T = MONT_MUL(U1, Hsq);

	uint256_g res_x = MONT_SQUARE(R);
	res_x = FIELD_SUB(res_x, Hcube);
	uint256_g temp = MONT_MUL(BASE_FIELD_R2, T);
	res_x = FIELD_SUB(res_x, temp);

	uint256_g res_y = FIELD_SUB(T, res_x);
	res_y = MONT_MUL(R, res_y);
	temp = MONT_MUL(S1, Hcube);
	res_y = FIELD_SUB(res_y, temp);

	uint256_g res_z = MONT_MUL(H, left.z);
	res_z = MONT_MUL(res_z, right.z);

	return ec_point{res_x, res_y, res_z};
}

DEVICE_FUNC ec_point ECC_SUB_JAC(const ec_point& left, const ec_point& right)
{
	return ECC_ADD_JAC(left, INV(right));
}

DEVICE_FUNC ec_point ECC_ADD_MIXED_JAC(const ec_point& left, const affine_point& right)
{
	if (is_infinity(left))
		return ec_point{right.x, right.y, BASE_FIELD_R};

	uint256_g U2;

	uint256_g Z1sq = MONT_SQUARE(left.z);
	U2 = MONT_MUL(right.x, Z1sq);
	
	uint256_g S2 = MONT_MUL(right.y, Z1sq);
	S2 = MONT_MUL(S2, left.z);

	if (EQUAL(left.x, U2))
	{
		if (!EQUAL(left.y, S2))
			return point_at_infty();
		else
			return  ECC_DOUBLE_JAC(left);
	}

	uint256_g H = FIELD_SUB(U2, left.x);
	uint256_g R = FIELD_SUB(S2, left.y);
	uint256_g Hsq = MONT_SQUARE(H);
	uint256_g Hcube = MONT_MUL(Hsq, H);
	uint256_g T = MONT_MUL(left.x, Hsq);

	uint256_g res_x = MONT_SQUARE(R);
	res_x = FIELD_SUB(res_x, Hcube);
	uint256_g temp = MONT_MUL(BASE_FIELD_R2, T);
	res_x = FIELD_SUB(res_x, temp);

	uint256_g res_y = FIELD_SUB(T, res_x);
	res_y = MONT_MUL(R, res_y);
	temp = MONT_MUL(left.y, Hcube);
	res_y = FIELD_SUB(res_y, temp);

	uint256_g res_z = MONT_MUL(H, left.z);

	return ec_point{res_x, res_y, res_z};
}

//TODO: what about repeated doubling (m-fold doubling) for Jacobian coordinates?

//random number generators
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

static DEVICE_FUNC inline uint256_g field_exp(const uint256_g& elem, const uint256_g& power)
{
    uint256_g S = elem;
	uint256_g Q = BASE_FIELD_R;

	for (size_t i = 0; i < N_BITLEN; i++)
	{
		bool flag = get_bit(power, i);
		if (flag)
        { 
            Q = MONT_MUL(Q, S);
        }
		
        S = MONT_SQUARE(S); 	
	}
	return Q;
}

//The following algorithm is taken from 1st edition of
//Jeffrey Hoffstein, Jill Pipher, J.H. Silverman - An introduction to mathematical cryptography
//Proposition 2.27 on page 84

static DEVICE_FUNC inline optional<uint256_g> field_square_root(const uint256_g& x)
{
    uint256_g candidate = field_exp(x, MAGIC_CONSTANT);

    using X = optional<uint256_g>;
    return (EQUAL(MONT_SQUARE(candidate), x) ? X(candidate) :  X(NONE_OPT));
}

DEVICE_FUNC void gen_random_elem(affine_point& pt, curandState& state)
{
	//consider equation in short Weierstrass form: y^2 = x^3 + a * x + b
    //generate random x and compute right hand side
    //if this is not a square - repeat, again and again, until we are successful
    uint256_g x;
    optional<uint256_g> y_opt;
    while (!y_opt)
    {
        gen_random_elem(x, state);

		//compute righthandside

		uint256_g righthandside = MONT_SQUARE(x);
		righthandside = MONT_MUL(righthandside, x);

		uint256_g temp = MONT_MUL(CURVE_A_COEFF, x);
		righthandside = FIELD_ADD(righthandside, temp);
		righthandside = FIELD_ADD(righthandside, CURVE_B_COEFF);

        y_opt = field_square_root(righthandside);
    }

    uint256_g y = y_opt.get_val();

    if (curand(&state) % 2)
        y = FIELD_ADD_INV(y);

    pt = affine_point{x, y};
}

DEVICE_FUNC void gen_random_elem(ec_point& pt, curandState& state)
{
	affine_point temp;
	gen_random_elem(temp, state);
	pt = ec_point{temp.x, temp.y, BASE_FIELD_R};

	//check if generated point is valid

	assert(IS_ON_CURVE(pt));
}

