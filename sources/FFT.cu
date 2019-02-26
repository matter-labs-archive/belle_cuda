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
//https://github.com/mmajko/FFT-cuda/blob/master/src/fft-cuda.cu

//http://eprints.utar.edu.my/2494/1/CS-2017-1401837-1.pdf

//NB: arr should be a power of two

__global__ void shuffle(uint256_g* __restrict__ input_arr, uint256_g* __restrict__ output_arr, uint32_t arr_len)
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
	uint256_g a;
	uint256_g b;
};

//array of precomputed roots of unity

DEVICE_FUNC field_pair fft_buttefly(const uint256_g& x, const uint256_g& y, const field& root_of_unity)
{
	field temp = y * root_of_unity;
	return field_pair{ x + temp, x - temp};
}


__device__ field get_root_of_unity(const field* arr, uint32_t index)
{
	field result(1);
	for (unsigned k = 0; k < 32; k++)
	{
		if (CHECK_BIT(index, k))
			result *= arr[k];
	}
	return result;
	
}

__global__ void fft_impl_iteration(field* __restrict__ input_arr, field* __restrict__ output_arr, const field* __restrict__ roots_of_unity_arr, 
	uint32_t arr_len, uint32_t log_arr_len, uint32_t step)
{
	uint32_t  i = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t k = (1 << step);
	uint32_t l = 2 * k;
	while (i < arr_len / 2)
	{
		uint32_t first_index = l * (i / k) + (i % k);
		uint32_t second_index = first_index + k;

		uint32_t root_of_unity_index = (1 <<(log_arr_len - step - 1)) * (i % l); 
		field omega = get_root_of_unity(roots_of_unity_arr, root_of_unity_index);

		field_pair ops = fft_buttefly(input_arr[first_index], input_arr[second_index], omega);

		output_arr[first_index] = ops.a;
		output_arr[second_index] = ops.b;

		i += blockDim.x * gridDim.x;
	}
}

void fft_impl(field* __restrict__ first_arr, field* __restrict__ second_arr, const field* __restrict__ roots_of_unity_arr,
	uint32_t arr_len, uint32_t log_arr_len)
{
	//shuffle(;
	
	for (uint32_t step = 0; step < log_arr_len; step++)
	{
		fft_impl_iteration<<<65535, 256>>>  (first_arr, second_arr, roots_of_unity_arr, arr_len, log_arr_len, step);

		field* temp = first_arr;
		first_arr = second_arr;
		second_arr = temp;
	}
}

#define FFT_N 1023
#define LOG_FFT_N 10

__constant__ field dev_roots_of_unity[LOG_FFT_N];


int main(int argc, char* argv[])
{
    bool result = CUDA_init() && crypto_init();

	if (!result)
	{
		printf("error");
		return -1;
	}
	
	//curve_gpu_test();
	//gpu_benchmark(&mont_mul_kernel_runner, 262144);
	//cpu_benchmark(&mont_mul_host, 262144);

	//exp_gpu_test();

	gpu_benchmark(&exp_kernel_runner, 10000);
	cpu_benchmark(&exp_host, 10000);
}

