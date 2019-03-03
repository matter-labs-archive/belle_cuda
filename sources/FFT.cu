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

	
__global__ void FFT_shuffle(embedded_field* __restrict__ input_arr, embedded_field* __restrict__ output_arr, uint32_t arr_len)
{
	uint32_t  tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		output_arr[tid] = input_arr[__brev(tid)];
		tid += blockDim.x * gridDim.x;
	}
}

struct field_pair
{
	embedded_field a;
	embedded_field b;
};

//array of precomputed roots of unity

DEVICE_FUNC field_pair fft_buttefly(const embedded_field& x, const embedded_field& y, const embedded_field& root_of_unity)
{
	embedded_field temp = y * root_of_unity;
	return field_pair{ x + temp, x - temp};
}

DEVICE_FUNC embedded_field get_root_of_unity(uint32_t index)
{
	embedded_field result(EMBEDDED_FIELD_R);
	for (unsigned k = 0; k < ROOTS_OF_UNTY_ARR_LEN; k++)
	{
		if (CHECK_BIT(index, k))
			result *= embedded_field(EMBEDDED_FIELD_ROOTS_OF_UNITY[k]);
	}
	return result;	
}

__global__ void FFT_iteration(embedded_field* __restrict__ input_arr, embedded_field* __restrict__ output_arr, 
	uint32_t arr_len, uint32_t log_arr_len, uint32_t step)
{
	uint32_t  i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t k = (1 << step);
	uint32_t l = 2 * k;
	while (i < arr_len / 2)
	{
		uint32_t first_index = l * (i / k) + (i % k);
		uint32_t second_index = first_index + k;

		uint32_t root_of_unity_index = (1 << (log_arr_len - step - 1)) * (i % l); 
		embedded_field omega = get_root_of_unity(root_of_unity_index);

		field_pair ops = fft_buttefly(input_arr[first_index], input_arr[second_index], omega);

		output_arr[first_index] = ops.a;
		output_arr[second_index] = ops.b;

		i += blockDim.x * gridDim.x;
	}
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

void fft_driver(embedded_field* __restrict__ input_arr, embedded_field* __restrict__ output_arr, uint32_t arr_len)
{
	//first check that arr_len is a power of 2

	uint log_arr_len = BITS_PER_LIMB - __builtin_clz(arr_len) - 1;
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
	FFT_shuffle<<<FFT_shuffle_geometry.gridSize, FFT_shuffle_geometry.blockSize>>>(input_arr, temp_output_arr, arr_len);
	
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

