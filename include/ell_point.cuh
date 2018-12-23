#ifndef ELL_POINT_CUH
#define ELL_POINT_CUH

#include "mont_mul.cuh"

//Again we initialize global variables with BN_256 specific values

DEVICE_VAR CONST_MEMORY uint256_g A_g = {
    0, 0, 0, 0, 0, 0, 0, 0
};

DEVICE_VAR CONST_MEMORY uint256_g B_g = {
    0x33e215dd,
    0x868d41a7,
    0xe1e724a2,
    0x937fcb76,
    0x374a1efb,
    0xdc782b83,
    0xd25b2206,
    0x11a4665b
};

struct ec_point
{
    uint256_g x;
    uint256_g y;
    uint256_g z;
};

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

//Arithmetic in projective coordinates (Jacobian coordinates should be faster and we are going to check it!)
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
		temp = FASTEST_montdouble_256(pt.x);
		W = FASTEST_montmul_256(temp, R3_g);
#else
		temp = FASTEST_montdouble_256(pt.x);
		temp = FASTEST_montmul_256(temp, R3_g);
		temp2 = FASTEST_montdouble_256(pt.z);
		temp2 = FASTEST_montmul_256(temp2, A_g);
		W = FASTEST_256_add(temp, temp2);
#endif
		S = FASTEST_montmul_256(pt.y, pt.z);
		temp = FASTEST_montmul_256(pt.x, pt.y);
		B = FASTEST_montmul_256(temp, S);

		temp = FASTEST_montdouble_256(W);
		temp2 = FASTEST_montmul_256(R8_g, B);
		H = FASTEST_256_sub(temp, temp2);

		temp = FASTEST_montmul_256(R2_g, H);
		res.x = FASTEST_montmul_256(temp, S);
		
		//NB: here result is also equal to one of the operands and hemce may be reused!!!

		S2 = FASTEST_montdouble_256(S);
		temp = FASTEST_montmul_256(4, B);
		temp = FASTEST_256_sub(temp, H);
		temp = FASTEST_montmul_256(W, temp);
		BASE_FIELD res_Y  = W * (mont_4 * B - H) - mont_8 * y_ * y_* S_squared;
		BASE_FIELD res_Z = mont_8 * S * S_squared;
		return EcPoint(res_X, res_Y, res_Z);
	}
}

	public:
		static HOST_AND_DEVICE_RUNNABLE EcPoint point_at_infty()
		{
			return EcPoint(0, 1, 0);
		}


		static bool init(const BASE_FIELD& a, const BASE_FIELD& b)
		{
			//copy curve params to constant memory
			cudaMemcpyToSymbol(dev_a_, &a, sizeof(BASE_FIELD));
			cudaMemcpyToSymbol(dev_b_, &b, sizeof(BASE_FIELD));

			auto rep = BASE_FIELD::modulus_;
			limb_index_t N = BASE_FIELD::n;

			//NB: the following code works on linux only!
			uint_default_t char_bitlen = sizeof(uint_default_t) * 8 * N;
			for (limb_index_t i = N; i > 0 ; i--)
			{
				if (rep[i - 1] == 0)
				{
					char_bitlen -= sizeof(uint_default_t) * 8;
				}
				else
				{
					uint_default_t clz = __builtin_clz(rep[i - 1]);
					char_bitlen -= clz;
					break;
				}
			}

			//std::cout << "char_bitlen = " << char_bitlen << std::endl;
			cudaMemcpyToSymbol(dev_char_bitlen_, &char_bitlen, sizeof(uint_default_t));

			return true;
		}

		//for debug purposes only: check if point is indeed on curve
		__device__ bool check_if_on_curve()
		{
			BASE_FIELD* a = reinterpret_cast<BASE_FIELD*>(&dev_a_);
			BASE_FIELD* b = reinterpret_cast<BASE_FIELD*>(&dev_b_);
			
			//y^{2} * z = x^{3} + a *x * z^{2} + b * z^{3}
			BASE_FIELD lefthand_side = y_ * y_ * z_;
			BASE_FIELD righthand_side = x_ * x_ * x_ + (*a) * x_ * z_ * z_ + (*b) * z_ * z_ * z_;
			return (lefthand_side == righthand_side);
		}

		//get random element: we use libff for this 
		//(cause we do not want to implement Tonelli-Schanks algorithm ourselves)
		//NB: it is now working in the case we use bn_256 curve, and in no other case
		static EcPoint get_random_elem()
		{
			return EcPoint(BASE_FIELD::get_random_elem(), BASE_FIELD::get_random_elem(), BASE_FIELD::get_random_elem());
		}

		//on host and device
		explicit HOST_AND_DEVICE_RUNNABLE EcPoint(const BASE_FIELD& x, const BASE_FIELD& y, const BASE_FIELD& z) : x_(x), y_(y), z_(z) {}
		explicit HOST_AND_DEVICE_RUNNABLE EcPoint() : x_(0), y_(1), z_(0) {}
		
		EcPoint(const EcPoint& other) = default;
		EcPoint(EcPoint&& other) = default;
		EcPoint& operator=(const EcPoint&) = default;
		EcPoint& operator=(EcPoint&&) = default;

		__device__ bool operator==(const EcPoint& other)
		{
			//check all of the following equations:
			//X_1 * Y_2 = Y_1 * X_2;
			//X_1 * Z_2 =  X_2 * Y_1;
			//Y_1 * Z_2 = Z_1 * Y_2;

			bool first_check = (x_ * other.y_ == y_ * other.x_);
			bool second_check = (y_ * other.z_ == z_ * other.y_);
			bool third_check = (x_ * other.z_ == z_ * other.x_);
			
			return (first_check && second_check && third_check);
		}

		__device__ bool operator!=(const EcPoint& other)
		{
			//check all of the following equations:
			//X_1 * Y_2 = Y_1 * X_2;
			//X_1 * Z_2 =  X_2 * Y_1;
			//Y_1 * Z_2 = Z_1 * Y_2;

			bool first_check = (x_ * other.y_ != y_ * other.x_);
			bool second_check = (y_ * other.z_ != z_ * other.y_);
			bool third_check = (x_ * other.z_ != z_ * other.x_);
			
			return (first_check || second_check || third_check);
		}

		__device__ EcPoint operator-()
		{
			return EcPoint(x_, -y_, z_);
		}
		
		__device__ EcPoint& operator+=(const EcPoint& other)
		{
			if (other.is_infty())
				return *this;
			if (this->is_infty())
			{
				*this = other;
				return *this;
			}
			
			BASE_FIELD U1 = z_ * other.y_;
			
			BASE_FIELD U2 = y_ * other.z_;
			BASE_FIELD V1 = z_ * other.x_;
			BASE_FIELD V2 = x_ * other.z_;

			//printf("V1: %x%x%x%x%x%x%x%x\n", V1.rep_[7], U1.rep_[6], U1.rep_[5], U1.rep_[4], U1.rep_[3], U1.rep_[2], U1.rep_[1], U1.rep_[0]);
			//printf("V2: %x%x%x%x%x%x%x%x\n", U2.rep_[7], U2.rep_[6], U2.rep_[5], U2.rep_[4], U2.rep_[3], U2.rep_[2], U2.rep_[1], U2.rep_[0]);

			if (V1 == V2)
			{
				if (U1 != U2)
				{
					*this = point_at_infty();					
				}
				else
				{
					*this = this->squared();
				}
			}
			else
			{
				BASE_FIELD& mont_2 = reinterpret_cast<BASE_FIELD&>(dev_mont_2_);
				
				BASE_FIELD U = U1 - U2;
				BASE_FIELD V = V1 - V2;
				BASE_FIELD W = z_ * other.z_;
				BASE_FIELD V_square = V * V;
				BASE_FIELD V_cube = V_square * V;
				BASE_FIELD A = U * U * W - V_cube - mont_2 * V_square * V2;
				BASE_FIELD X3 = V * A;
				BASE_FIELD Y3 = U * (V_square * V2 - A) - V_cube * U2;
				BASE_FIELD Z3 = V_cube * W;

				*this = EcPoint(X3, Y3, Z3);
			}

			return *this;
		}

		__device__ EcPoint& operator-=(const EcPoint& other)
		{
			*this += (-other);
			return *this;
		}

		//exponentiation
		__device__ EcPoint operator^(const BASE_FIELD& power)
		{
			//use classical double and add algorithm
			//NB: there is a slightly more efficient algorithm using ternary expansion

			EcPoint N = *this;
			EcPoint Q = point_at_infty();
			for (size_t i = 0; i < dev_char_bitlen_; i++)
			{
				bool flag = power.rep_.get_bit(i);
				if (flag)
				{
					Q += N;	
				}
				N = N.squared();			
			}
			return Q;
		}

		template<typename BASE_FIELD2, typename curve_tag2>
		friend EcPoint<BASE_FIELD2, curve_tag2> operator+(const EcPoint<BASE_FIELD2, curve_tag2>& left, const EcPoint<BASE_FIELD2, curve_tag2>& right);
		template<typename BASE_FIELD2, typename curve_tag2>
		friend EcPoint<BASE_FIELD2, curve_tag2> operator-(const EcPoint<BASE_FIELD2, curve_tag2>& left, const EcPoint<BASE_FIELD2, curve_tag2>& right);
		template<typename BASE_FIELD2, typename curve_tag2>
		friend std::ostream& operator<<(std::ostream& os, const EcPoint<BASE_FIELD2, curve_tag2>& num);
	};

	template<typename BASE_FIELD, typename curve_tag>
	__device__ EcPoint<BASE_FIELD, curve_tag> operator+(const EcPoint<BASE_FIELD, curve_tag>& left, const EcPoint<BASE_FIELD, curve_tag>& right)
	{
		EcPoint<BASE_FIELD, curve_tag> result(left);
		result += right;
		return result;
	}

	template<typename BASE_FIELD, typename curve_tag>
	__device__ EcPoint<BASE_FIELD, curve_tag> operator-(const EcPoint<BASE_FIELD, curve_tag>& left, const EcPoint<BASE_FIELD, curve_tag>& right)
	{
		EcPoint<BASE_FIELD, curve_tag> result(left);
		result -= right;
		return result;
	}

	template<typename BASE_FIELD, typename elliptic_curve_tag>
	std::ostream& operator<<(std::ostream& os, const EcPoint<BASE_FIELD, elliptic_curve_tag>& elem)
	{
		os << "EC_POINT: " << std::endl;
		os << "x = " << elem.x_  << std::endl;
		os << "y = " << elem.y_  << std::endl;
		os << "z = " << elem.z_  << std::endl;
		return os;
	}
};

#endif

#endif