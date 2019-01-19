#include "cuda_structs.h"
#include <stdio.h>

#include <type_traits>
#include <stdexcept>

#include <iostream>
#include <iomanip>

static constexpr uint256_g MODULUS = {
    0xd87cfd47,
    0x3c208c16,
    0x6871ca8d,
    0x97816a91,
    0x8181585d,
    0xb85045b6,
    0xe131a029,
    0x30644e72 
};

 static constexpr uint256_g R = 
 {
    0xc58f0d9d,
    0xd35d438d,
    0xf5c70b3d,
    0x0a78eb28,
    0x7879462c,
    0x666ea36f,
    0x9a07df2f,
    0xe0a77c1
 };

static constexpr uint32_t N_INV = 0xe4866389;

//this file preimarly contains code for generating random point on BN-curve

inline uint256_g add_uint256_host(const uint256_g& lhs, const uint256_g& rhs)
{
    uint256_g result;
    uint32_t carry = 0;

    for (uint32_t i = 0; i < N; i++)
    {
        result.n[i] = lhs.n[i] + rhs.n[i] + carry;
        carry = (result.n[i] < lhs.n[i]);
    }
    return result;
}


inline uint256_g sub_uint256_host(const uint256_g& lhs, const uint256_g& rhs)
{
    uint32_t borrow = 0;
    uint256_g result;
    
	for (uint32_t i = 0; i < N; i++)   
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


inline int cmp_uint256_host(const uint256_g& lhs, const uint256_g& rhs)
{
    for (int32_t i = N -1 ; i >= 0; i--)
    {
        if (lhs.n[i] > rhs.n[i])
            return 1;
        else if (lhs.n[i] < rhs.n[i])
            return -1;
    }
    return 0;
}

inline bool is_zero_host(const uint256_g& x)
{
    for (int32_t i = 0 ; i < N; i++)
    {
        if (x.n[i] != 0)
            return false;
    }
    return true;
}


inline uint32_t host_long_mul(uint32_t x, uint32_t y, uint32_t* high_ptr)
	{
		uint64_t res = (uint64_t)x * (uint64_t)y;
		*high_ptr = (res >> 32);
		return res;
	}

inline uint32_t host_fused_add(uint32_t x, uint32_t y, uint32_t* high_ptr)
{
	uint32_t z = x + y;
	if (z < x)
		(*high_ptr)++;
    return z;
}


inline uint256_g mont_mul_256_host(const uint256_g& u, const uint256_g& v)
{
    uint256_g T;

	for (uint32_t j = 0; j < N; j++)
        T.n[j] = 0;

    uint32_t prefix_low = 0, prefix_high = 0, m;
    uint32_t high_word, low_word;

    for (uint32_t i = 0; i < N; i++)
    {
        uint32_t carry = 0;
        for (uint32_t j = 0; j < N; j++)
        {         
            low_word = host_long_mul(u.n[j], v.n[i], &high_word);
            low_word = host_fused_add(low_word, T.n[j], &high_word);
            low_word = host_fused_add(low_word, carry, &high_word);
            carry = high_word;
            T.n[j] = low_word;
        }

        //TODO: may be we actually require less space? (only one additional limb instead of two)
        prefix_high = 0;
        prefix_low = host_fused_add(prefix_low, carry, &prefix_high);

        m = T.n[0] * N_INV;
        low_word = host_long_mul(MODULUS.n[0], m, &high_word);
        low_word = host_fused_add(low_word, T.n[0], &high_word);
        carry = high_word;

        #pragma unroll
        for (uint32_t j = 1; j < N; j++)
        {
            low_word = host_long_mul(MODULUS.n[j], m, &high_word);
            low_word = host_fused_add(low_word, T.n[j], &high_word);
            low_word = host_fused_add(low_word, carry, &high_word);
            T.n[j-1] = low_word;
            carry = high_word;
        }

        T.n[N-1] = host_fused_add(prefix_low, carry, &prefix_high);
        prefix_low = prefix_high;
    }
    
    if (cmp_uint256_host(T, MODULUS) >= 0)
    {
        //TODO: may be better change to inary version of sub?
        T = sub_uint256_host(T, MODULUS);
    }

    return T;
}

//It's safe: cause we are going to use this class only on the host

class Field
{
private:
    uint256_g rep_;
public:
    static Field zero()
    {
        return Field(0);
    }

    Field(uint32_t n = 0)
    {
        for (size_t i = 1; i < N; i++)
            rep_.n[i] = 0;
        rep_.n[0] = n;
    }

    explicit Field(uint256_g n)
    {
        for (size_t i = 0; i < N; i++)
            rep_.n[i] = n.n[i];
    }
   
    Field(const Field& other) = default;
    Field(Field&& other) = default;
    Field& operator=(const Field&) = default;
    Field& operator=(Field&&) = default;

    bool operator==(const Field& other) const
    {
        return cmp_uint256_host(rep_, other.rep_) == 0;
    }

    bool operator!=(const Field& other) const
    {
        return cmp_uint256_host(rep_, other.rep_) != 0;
    }

    Field operator-()
    {
        uint256_g ans = (is_zero_host(rep_) ? zero().rep_ : sub_uint256_host(MODULUS, rep_));
        return Field(ans);
    }

    //NB: for now we assume that highest possible limb bit is zero for the field modulus
    Field& operator+=(const Field& other)
    {
        rep_ = add_uint256_host(rep_, other.rep_);
        if (cmp_uint256_host(rep_, MODULUS) >= 0)
            rep_ = sub_uint256_host(rep_, MODULUS);
        return *this;
    }

    Field& operator-=(const Field& other)
    {
        if (cmp_uint256_host(rep_, other.rep_) < 0)
             rep_ = add_uint256_host(rep_, MODULUS);
        rep_ = sub_uint256_host(rep_, other.rep_);
        return *this;
    }

    //here we mean montgomery multiplication
    Field& operator*=(const Field& other)
    {
        rep_ = mont_mul_256_host(rep_, other.rep_);
        return *this;
    }

    uint256_g get_raw_rep() const
    {
        return rep_;
    }

    friend Field operator+(const Field& left, const Field& right);
    friend Field operator-(const Field& left, const Field& right);
    friend Field operator*(const Field& left, const Field& right);

	friend std::ostream& operator<<(std::ostream& os, const Field& elem);
};

Field operator+(const Field& left, const Field& right)
{
    Field result(left);
    result += right;
    return result;
}

Field operator-(const Field& left, const Field& right)
{
    Field result(left);
    result -= right;
    return result;
}

Field operator*(const Field& left, const Field& right)
{
    Field result(left);
    result *= right;
    return result;
}

std::ostream& operator<<(std::ostream& os, const Field& elem)
{
    os << "0x";
    for (int i = 7; i >= 0; i--)
    {
        os << std::setfill('0') << std::hex << std::setw(8) << elem.rep_.n[i];
    }
    return os;
}


inline bool get_bit_host(const uint256_g& x, size_t index)
{
	auto num = x.n[index / 32];
	auto pos = index % 32;
	return CHECK_BIT(num, pos);
}

inline Field exp_host(const Field& elem, const uint256_g& power)
{
    Field S = elem;
	Field Q = Field(R);

	for (size_t i = 0; i < N_BITLEN; i++)
	{
		bool flag = get_bit_host(power, i);
		if (flag)
        { 
            Q *= S;
        }
		
        S *= S; 	
	}
	return Q;
}

//We are not able to compile with C++ 17 standard

struct none_t{};
static constexpr none_t NONE_OPT;

template<typename T>
class optional
{
private:
    bool flag_;
    T val_;

    static_assert(std::is_default_constructible<T>::value, "Inner type of optional should be constructible!");
public:
    optional(const T& val): flag_(true), val_(val) {}
    optional(const none_t& none): flag_(false) {}
    optional(): flag_(false) {}

    optional(const optional& other) = default;
    optional(optional&& other) = default;
    optional& operator=(const optional&) = default;
    optional& operator=(optional&&) = default;

    operator bool() const
    {
        return flag_;
    }

    const T& get_val() const
    {
        if (!flag_)
            throw std::runtime_error("Retrieving value of empty optional!");
        return val_;
    } 
};

//The following algorithm is taken from 1st edition of
//Jeffrey Hoffstein, Jill Pipher, J.H. Silverman - An introduction to mathematical cryptography
//Proposition 2.27 on page 84
//if p = 3 (mod 4) , what is true for BN256-curve underlying field, and x^2 = a is satisfyable, then
//x = a ^ (p + 1)/4
//NB: the equation x^2 = a may have no solutions at all, so after computing x we require to check that it'is indeed a solution
//NB: MAGIC_POWER =(P+1)/4 is constant, so we are able to precompute it
//NB: Magic constant should be given in standard form (i.e. NON MONTGOMERY)

static constexpr uint256_g MAGIC_CONSTANT = 
{
    0xb61f3f52,
    0x4f082305,
    0x5a1c72a3,
    0x65e05aa4,
    0xa0605617,
    0x6e14116d,
    0xb84c680a,
    0xc19139c 
};

optional<Field> square_root_host(const Field& x)
{
    Field candidate = exp_host(x, MAGIC_CONSTANT);

    using X = optional<Field>;
    return (candidate * candidate == x ? X(candidate) :  X(NONE_OPT));
}

//NB: we don't need to check that our random point does belong to the right subgroup:
//more precisely to that one, generated by  G = [1, 2, 1]
//this is because cofactor is 1!

//NB: in our elliptic curve a = 0, b = 3, but we are working in montgomery form, so these coefficients are also taken in montgomery form
//NB: after getting x, y the coordinate of point in projective: [x, y, mont(1)], and the same for jacobian!

static constexpr uint256_g A = {
    0, 0, 0, 0, 0, 0, 0, 0
};

static constexpr uint256_g B = 
{
     0x50ad28d7,
     0x7a17caa9,
     0xe15521b9,
     0x1f6ac17a,
     0x696bd284,
     0x334bea4e,
     0xce179d8e,
     0x2a1f6744 
 };


Field get_random_field_elem()
{
    uint256_g res;
    for (uint32_t i =0; i < N; i++)
        res.n[i] = rand();
    res.n[N - 1] &= 0x1fffffff;
    return Field(res);
}


ec_point get_random_point_host()
{
    //equation in Weierstrass form: y^2 = x^3 + a * x + b
    //generate random x and compute right hand side
    //if this is not a square - repeat, again and again, until we are successful
    Field x;
    optional<Field> y_opt;
    while (!y_opt)
    {
        x = get_random_field_elem();
        Field righthandside = x * x * x + Field(A) * x + Field(B);
        y_opt = square_root_host(righthandside);
    }

    Field y = y_opt.get_val();

    if (rand() % 2)
        y = -y;

    return ec_point{x.get_raw_rep(), y.get_raw_rep(), R};
}

