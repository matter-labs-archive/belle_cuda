#include "cuda_structs.h"

//FFT (we propose very naive realization)
//----------------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------------

//Sources of inspiration:
//http://www.staff.science.uu.nl/~bisse101/Articles/preprint1138.pdf
//https://cs.wmich.edu/gupta/teaching/cs5260/5260Sp15web/studentProjects/tiba&hussein/03278999.pdf
//http://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/DeSpain_FFT_Presentation.pdf
//http://www.bealto.com/gpu-fft_intro.html
//https://github.com/mmajko/FFT-cuda/blob/master/src/fft-cuda.cu
//Also have a loot at GPU gems

//NB: arr should be a power of two

//this is a field embedded into a group of points on elliptic curve

//because of FUNCKING NVIDIA CREW I'am unable to use inline asm here

struct embedded_field
{
	uint256_g rep_;

	DEVICE_FUNC explicit embedded_field(const uint256_g rep): rep_(rep) {}
	
	DEVICE_FUNC bool operator==(const embedded_field& other) const
	{
		return EQUAL(rep_, other.rep_);
	}

	static DEVICE_FUNC embedded_field zero()
	{
		uint256_g x;
		#pragma unroll
		for(uint32_t i = 0; i < N; i++)
			x.n[i] = 0;

		return embedded_field(x);
	}

	static DEVICE_FUNC embedded_field one()
	{
		return embedded_field(EMBEDDED_FIELD_R);
	}

	DEVICE_FUNC bool operator!=(const embedded_field& other) const
	{
		return !EQUAL(rep_, other.rep_);
	}

	DEVICE_FUNC operator uint256_g() const
	{
		return rep_;
	}

	DEVICE_FUNC embedded_field operator-() const
	{
		if (!is_zero(rep_))
			return embedded_field(SUB(EMBEDDED_FIELD_P, rep_));
		else
			return *this;
	}

	//NB: for now we assume that highest possible limb bit is zero for the field modulus
	DEVICE_FUNC embedded_field& operator+=(const embedded_field& other)
	{
		rep_ = ADD(rep_, other.rep_);
		if (CMP(rep_, EMBEDDED_FIELD_P) >= 0)
			rep_ = SUB(rep_, EMBEDDED_FIELD_P);
		return *this;
	}

	DEVICE_FUNC embedded_field& operator-=(const embedded_field& other)
	{
		if (CMP(rep_, other.rep_) > 0)
			rep_ = SUB(rep_, other.rep_);
		else
		{
			uint256_g t = ADD(rep_, EMBEDDED_FIELD_P);
			rep_ = SUB(t, other.rep_);
		}
		return *this;
	}

	//here we mean montgomery multiplication

	DEVICE_FUNC embedded_field& operator*=(const embedded_field& other)
	{
		uint256_g T;
		uint256_g u = rep_;
		uint256_g v = other.rep_;

		#pragma unroll
		for (uint32_t j = 0; j < N; j++)
			T.n[j] = 0;

		uint32_t prefix_low = 0, prefix_high = 0, m;
		uint32_t high_word, low_word;

		#pragma unroll
		for (uint32_t i = 0; i < N; i++)
		{
			uint32_t carry = 0;
			#pragma unroll
			for (uint32_t j = 0; j < N; j++)
			{         
				low_word = device_long_mul(u.n[j], v.n[i], &high_word);
				low_word = device_fused_add(low_word, T.n[j], &high_word);
				low_word = device_fused_add(low_word, carry, &high_word);
				carry = high_word;
				T.n[j] = low_word;
			}

			//TODO: may be we actually require less space? (only one additional limb instead of two)
			prefix_high = 0;
			prefix_low = device_fused_add(prefix_low, carry, &prefix_high);

			m = T.n[0] * EMBEDDED_FIELD_N;
			low_word = device_long_mul(EMBEDDED_FIELD_P.n[0], m, &high_word);
			low_word = device_fused_add(low_word, T.n[0], &high_word);
			carry = high_word;

			#pragma unroll
			for (uint32_t j = 1; j < N; j++)
			{
				low_word = device_long_mul(EMBEDDED_FIELD_P.n[j], m, &high_word);
				low_word = device_fused_add(low_word, T.n[j], &high_word);
				low_word = device_fused_add(low_word, carry, &high_word);
				T.n[j-1] = low_word;
				carry = high_word;
			}

			T.n[N-1] = device_fused_add(prefix_low, carry, &prefix_high);
			prefix_low = prefix_high;
		}
		
		if (CMP(T, EMBEDDED_FIELD_P) >= 0)
		{
			//TODO: may be better change to inary version of sub?
			T = SUB(T, EMBEDDED_FIELD_P);
		}

		rep_ = T;
		return *this;
	}
		
	friend DEVICE_FUNC embedded_field operator+(const embedded_field& left, const embedded_field& right);
	friend DEVICE_FUNC embedded_field operator-(const embedded_field& left, const embedded_field& right);
	friend DEVICE_FUNC embedded_field operator*(const embedded_field& left, const embedded_field& right);
};
	

DEVICE_FUNC embedded_field operator+(const embedded_field& left, const embedded_field& right)
{
	embedded_field result(left);
	result += right;
	return result;
}

DEVICE_FUNC embedded_field operator-(const embedded_field& left, const embedded_field& right)
{
	embedded_field result(left);
	result -= right;
	return result;
}

DEVICE_FUNC embedded_field operator*(const embedded_field& left, const embedded_field& right)
{
	embedded_field result(left);
	result *= right;
	return result;
}


//commom FFT routines
//------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------------

struct field_pair
{
	embedded_field a;
	embedded_field b;
};

DEVICE_FUNC field_pair __inline__ fft_buttefly(const embedded_field& x, const embedded_field& y, const embedded_field& root_of_unity)
{
	embedded_field temp = y * root_of_unity;
	return field_pair{ x + temp, x - temp};
}

DEVICE_FUNC embedded_field __inline__ get_root_of_unity(uint32_t index, uint32_t omega_idx_coeff = 1, bool inverse = false)
{
	embedded_field result(EMBEDDED_FIELD_R);
	uint32_t real_idx = index * omega_idx_coeff;
	if (inverse)
		real_idx = (1 << ROOTS_OF_UNTY_ARR_LEN) - real_idx;
	for (unsigned k = 0; k < ROOTS_OF_UNTY_ARR_LEN; k++)
	{
		if (CHECK_BIT(real_idx, k))
			result *= embedded_field(EMBEDDED_FIELD_ROOTS_OF_UNITY[k]);
	}
	return result;	
}

struct geometry
{
    int gridSize;
    int blockSize;
};

template<typename T>
geometry find_suitable_geometry(T func, uint shared_memory_used, uint32_t smCount)
{
    int gridSize;
    int blockSize;
    int maxActiveBlocks;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, shared_memory_used, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, shared_memory_used);
    gridSize = maxActiveBlocks * smCount;

    return geometry{gridSize, blockSize};
}

//Naive FFT-realization
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void FFT_shuffle(embedded_field* __restrict__ input_arr, embedded_field* __restrict__ output_arr, uint32_t arr_len, uint32_t log_arr_len)
{
	uint32_t  tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		output_arr[tid] = input_arr[__brev(tid) >> (32 - log_arr_len)];
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void FFT_iteration(embedded_field* __restrict__ input_arr, embedded_field* __restrict__ output_arr, 
	uint32_t arr_len, uint32_t log_arr_len, uint32_t step)
{
	uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t k = (1 << step);
	uint32_t l = 2 * k;
	size_t omega_coeff = 1 << (ROOTS_OF_UNTY_ARR_LEN - log_arr_len);
	while (i < arr_len / 2)
	{
		uint32_t first_index = l * (i / k) + (i % k);
		uint32_t second_index = first_index + k;

		uint32_t root_of_unity_index = (1 << (log_arr_len - step - 1)) * (i % k); 
		embedded_field omega = get_root_of_unity(root_of_unity_index, omega_coeff);

		field_pair ops = fft_buttefly(input_arr[first_index], input_arr[second_index], omega);

		output_arr[first_index] = ops.a;
		output_arr[second_index] = ops.b;

		i += blockDim.x * gridDim.x;
	}
}

#include <iostream>

void naive_fft_driver(embedded_field* input_arr, embedded_field* output_arr, uint32_t arr_len, bool is_inverse_FFT = false)
{
	//first check that arr_len is a power of 2

	uint log_arr_len = BITS_PER_LIMB - __builtin_clz(arr_len) - 1;
	std::cout << "Log arr len: " << log_arr_len << std::endl;
    assert(arr_len = (1 << log_arr_len));

	//find optimal geometry

	cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

	geometry FFT_shuffle_geometry = find_suitable_geometry(FFT_shuffle, 0, smCount);
	geometry FFT_iter_geometry = find_suitable_geometry(FFT_iteration, 0, smCount);

	//allocate additional memory

	embedded_field* additional_device_memory = nullptr;
	cudaError_t cudaStatus = cudaMalloc((void **)&additional_device_memory, arr_len * sizeof(embedded_field));
	
	//FFT shuffle;

	embedded_field* temp_output_arr = (log_arr_len % 2 ? additional_device_memory : output_arr);
	embedded_field* temp_input_arr = (log_arr_len % 2 ? output_arr : additional_device_memory);
	FFT_shuffle<<<FFT_shuffle_geometry.gridSize, FFT_shuffle_geometry.blockSize>>>(input_arr, temp_output_arr, arr_len, log_arr_len);
	
	//FFT main cycle

	for (uint32_t step = 0; step < log_arr_len; step++)
	{
		//swap input and iutput arrs

		embedded_field* swap_arr = temp_input_arr;
		temp_input_arr = temp_output_arr;
		temp_output_arr = swap_arr;
		
		FFT_iteration<<<FFT_iter_geometry.gridSize, FFT_iter_geometry.blockSize>>>(temp_input_arr, temp_output_arr, arr_len, log_arr_len, step);
	}

	//clean_up
	cudaFree(additional_device_memory);
}


//Bellman FFT-realization
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------

//TODO: make the same things using shuffle instructions

DEVICE_FUNC void _basic_serial_radix2_FFT(embedded_field* arr, size_t log_arr_len, size_t omega_idx_coeff, bool is_inverse_FFT)
{
	size_t tid = threadIdx.x;
	size_t arr_len = 1 << log_arr_len;

	for(size_t i = tid; i < arr_len; i+= blockDim.x)
	{
		size_t rk = __brev(i);
		if (i < rk)
		{	
			embedded_field temp = arr[i];
			arr[i] = arr[rk];
			arr[rk] = temp;
		}
	}

	__syncthreads();
	
    for (size_t step = 1; step <= log_arr_len; ++step)
    {
        uint32_t i = tid;
		uint32_t k = (1 << step);
		uint32_t l = 2 * k;
		while (i < arr_len / 2)
		{
			uint32_t first_index = l * (i / k) + (i % k);
			uint32_t second_index = first_index + k;

			uint32_t omega_idx = (1 << (log_arr_len - step - 1)) * (i % l); 
			embedded_field omega = get_root_of_unity(omega_idx, omega_idx_coeff, is_inverse_FFT);

			field_pair ops = fft_buttefly(arr[first_index], arr[second_index], omega);

			arr[first_index] = ops.a;
			arr[second_index] = ops.b;

			i += blockDim.x;
		}
		
		__syncthreads();
	}
}

__global__ void _basic_parallel_radix2_FFT(const embedded_field* input_arr, embedded_field* output_arr, size_t log_arr_len, size_t log_num_subblocks,
	bool is_inverse_FFT)
{
    extern __shared__ embedded_field temp_arr[];

	assert( log_arr_len <= ROOTS_OF_UNTY_ARR_LEN && "the size of array is too large for FFT");

	size_t omega_coeff = 1 << (ROOTS_OF_UNTY_ARR_LEN - log_arr_len);
	size_t L = 1 << (log_arr_len - log_num_subblocks);
	size_t NUM_SUBBLOCKS = 1 << log_num_subblocks;

	embedded_field omega_step = get_root_of_unity(blockIdx.x * L, omega_coeff, is_inverse_FFT);
        
    for (size_t i = threadIdx.x; i < L; i+= blockDim.x)
    {
        embedded_field omega_init = get_root_of_unity(blockIdx.x * threadIdx.x, omega_coeff, is_inverse_FFT);
		temp_arr[i] = embedded_field::zero();
		for (size_t s = 0; s < NUM_SUBBLOCKS; ++s)
        {
            size_t idx = i + s * L;
            temp_arr[i] += input_arr[idx] * omega_init;
            omega_init *= omega_step;
        }
	}

	__syncthreads();

	_basic_serial_radix2_FFT(temp_arr, log_arr_len, NUM_SUBBLOCKS, is_inverse_FFT);

	for (size_t i = threadIdx.x; i < L; i+= blockDim.x)
		output_arr[i * NUM_SUBBLOCKS + blockIdx.x] = temp_arr[i];
}

geometry find_geometry_for_advanced_FFT(uint log_arr_len)
{
	geometry res;
	
	//TODO: apply some heuristics!
	uint32_t x = log_arr_len >> 1;
	res.gridSize = (1 << x);
	res.blockSize = (1 << (x - 1));
	return res;
}

void advanced_fft_driver(embedded_field* input_arr, embedded_field* output_arr, uint32_t arr_len, bool is_inverse_FFT = false)
{
	//first check that arr_len is a power of 2

	uint log_arr_len = BITS_PER_LIMB - __builtin_clz(arr_len) - 1;
    assert(arr_len = (1 << log_arr_len));

	//find optimal geometry

	// cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
	// uint32_t smCount = prop.multiProcessorCount;

	geometry kernel_geometry = find_geometry_for_advanced_FFT(log_arr_len);
	uint log_num_subblocks = BITS_PER_LIMB - __builtin_clz(kernel_geometry.blockSize) - 1;

	_basic_parallel_radix2_FFT<<<kernel_geometry.gridSize, kernel_geometry.blockSize, kernel_geometry.blockSize * 2>>>(input_arr, output_arr, 
		log_arr_len, log_num_subblocks, is_inverse_FFT);
	cudaDeviceSynchronize();
}

#define FFT_DRIVER(input_arr, output_arr, arr_len, is_inverse_FFT) advanced_fft_driver(input_arr, output_arr, arr_len, is_inverse_FFT)


//polynomial multiplication via FFT

struct polynomial
{
	size_t deg;
	embedded_field* coeffs;
};

size_t get_power_of_two(size_t n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    return n;
}

__global__ void _mul_vecs(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = (a_arr[tid] * b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

void _mul_vecs_driver(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, size_t arr_len)
{
	int blockSize;
  	int minGridSize;
  	int realGridSize;
	int maxActiveBlocks;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, _mul_vecs, 0, 0);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;

	cudaDeviceProp prop;
  	cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;
	cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, _mul_vecs, blockSize, 0);
    if (error == cudaSuccess)
    	realGridSize = maxActiveBlocks * smCount;

	_mul_vecs<<<realGridSize, blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

polynomial _polynomial_multiplication_on_fft(const polynomial& A, const polynomial& B)
{
    size_t n = get_power_of_two(A.deg + B.deg);
	polynomial C;
	C.deg = A.deg + B.deg;

	embedded_field* temp_memory1 = nullptr;
	embedded_field* temp_memory2 = nullptr;
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void **)&temp_memory1, n * sizeof(embedded_field));
	cudaStatus = cudaMalloc((void **)&temp_memory2, n * sizeof(embedded_field));
	cudaStatus = cudaMalloc((void **)&C.coeffs, n * sizeof(embedded_field));
	
	cudaMemcpy(temp_memory1, A.coeffs, A.deg * sizeof(embedded_field), cudaMemcpyDeviceToDevice);
	cudaMemset(temp_memory1 + A.deg, 0, (n - A.deg) *sizeof(embedded_field));
	cudaMemcpy(temp_memory2, B.coeffs, B.deg * sizeof(embedded_field), cudaMemcpyDeviceToDevice);
	cudaMemset(temp_memory2 + B.deg, 0, (n - B.deg) *sizeof(embedded_field));

    FFT_DRIVER(temp_memory1, temp_memory1, n, false);
	FFT_DRIVER(temp_memory2, temp_memory2, n, false);
	
	_mul_vecs_driver(temp_memory1, temp_memory2, C.coeffs, n);	
	FFT_DRIVER(C.coeffs, C.coeffs, n, true);
	//_mul_elem_driver(C.coeffs, get_inv(n), n);

	cudaFree(temp_memory1);
	cudaFree(temp_memory2);

	return C;
}

#define POLY_MUL(X, Y) _polynomial_multiplication_on_fft(X, Y)


//these drivers are used only for test purposes
//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

void naive_FFT_test_driver(uint256_g* A, uint256_g* B, uint256_g* C, size_t arr_len)
{
	naive_fft_driver(reinterpret_cast<embedded_field*>(A), reinterpret_cast<embedded_field*>(C), arr_len);
}

void advanced_fft_test_driver(uint256_g* A, uint256_g* B, uint256_g* C, size_t arr_len)
{
	advanced_fft_driver(reinterpret_cast<embedded_field*>(A), reinterpret_cast<embedded_field*>(C), arr_len);
}
