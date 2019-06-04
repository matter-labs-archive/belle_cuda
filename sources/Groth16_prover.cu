#include "cuda_structs.h"

#include <iostream>
#include <sys/mman.h>

#define EXPORT __attribute__((visibility("default")))

//----------------------------------------------------------------------------------------------------------------------------------------------
//In order to simplify testing we will have all required functions in one file (NB: it leads to awful duplicated in code, but who cares!) 
//----------------------------------------------------------------------------------------------------------------------------------------------

struct Geometry
{
    int gridSize;
    int blockSize;
};

template<typename T>
Geometry find_suitable_geometry(T func, uint shared_memory_used, uint32_t smCount)
{
    int gridSize;
    int blockSize;
    int maxActiveBlocks;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, shared_memory_used, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, shared_memory_used);
    gridSize = maxActiveBlocks * smCount;

    return Geometry{gridSize, blockSize};
}

static void HandleError(cudaError_t err, const char *file, int line )
{
    if (err != cudaSuccess)
    {
        std::cout << cudaGetErrorString( err ) << " in " << file << " at line " << line << std::endl;
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__constant__ uint256_g elems[3];
__constant__ uint256_g tau[1];

//----------------------------------------------------------------------------------------------------------------------------------------------
//vector operations
//----------------------------------------------------------------------------------------------------------------------------------------------

__global__ void field_sub_inplace_kernel(embedded_field* a_arr, const embedded_field* b_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		a_arr[tid] -= b_arr[tid];
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void field_mul_inplace_kernel(embedded_field* a_arr, const embedded_field* b_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		a_arr[tid] *= b_arr[tid];
		tid += blockDim.x * gridDim.x;
	}
}

using field_kernel_t = void(embedded_field*, const embedded_field*, size_t); 

void field_func_invoke(embedded_field* a_arr, const embedded_field* b_arr, uint32_t arr_len, cudaStream_t& stream, 
    uint32_t smCount, field_kernel_t func)
{
    Geometry geometry = find_suitable_geometry(func, 0, smCount);

    (*func)<<<geometry.gridSize, geometry.blockSize, 0, stream>>>(a_arr, b_arr, arr_len);
}

__global__ void field_fused_mul_sub_inplace_kernel(embedded_field* a_arr, const embedded_field* b_arr, 
    const embedded_field* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		a_arr[tid] *= b_arr[tid];
        a_arr[tid] -= c_arr[tid];
		tid += blockDim.x * gridDim.x;
	}
}

void fused_mul_sub(embedded_field* a_arr, const embedded_field* b_arr, const embedded_field* c_arr, uint32_t arr_len, uint32_t smCount)
{
    Geometry geometry = find_suitable_geometry(field_fused_mul_sub_inplace_kernel, 0, smCount);
    field_fused_mul_sub_inplace_kernel<<<geometry.gridSize, geometry.blockSize>>>(a_arr, b_arr, c_arr, arr_len);
}

//----------------------------------------------------------------------------------------------------------------------------------------------
//FFT
//----------------------------------------------------------------------------------------------------------------------------------------------

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
		real_idx = (1 << (ROOTS_OF_UNTY_ARR_LEN)) - real_idx;
	for (unsigned k = 0; k < ROOTS_OF_UNTY_ARR_LEN; k++)
	{
		if (CHECK_BIT(real_idx, k))
			result *= embedded_field(EMBEDDED_FIELD_ROOTS_OF_UNITY[k]);
	}
	return result;	
}

__global__ void fft_shuffle(embedded_field* arr, uint32_t arr_len, uint32_t log_arr_len)
{
	uint32_t  tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		uint32_t first_idx = tid;
        uint32_t second_idx = __brev(first_idx) >> (32 - log_arr_len);
        if (first_idx  < second_idx)
        {
            //swap values!
            embedded_field temp = arr[first_idx];
            arr[first_idx] = arr[second_idx];
            arr[second_idx] = temp;
        }
    
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void fft_iteration(embedded_field* arr, uint32_t arr_len, uint32_t log_arr_len, uint32_t step, bool is_inverse)
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
		embedded_field omega = get_root_of_unity(root_of_unity_index, omega_coeff, is_inverse);

		field_pair ops = fft_buttefly(arr[first_index], arr[second_index], omega);

		arr[first_index] = ops.a;
		arr[second_index] = ops.b;

		i += blockDim.x * gridDim.x;
	}
}

void fft_impl(embedded_field* arr, uint32_t arr_len, bool is_inverse_FFT, const Geometry& geometry, cudaStream_t& stream)
{
	uint log_arr_len = BITS_PER_LIMB - __builtin_clz(arr_len) - 1;
	fft_shuffle<<<geometry.gridSize, geometry.blockSize, 0, stream>>>(arr, arr_len, log_arr_len);
	
	//FFT main cycle
	for (uint32_t step = 0; step < log_arr_len; step++)
	{
		fft_iteration<<<geometry.gridSize, geometry.blockSize, 0, stream>>>(arr, arr_len, log_arr_len, step, is_inverse_FFT);
	}
}


__global__ void mul_by_const_kernel(embedded_field* arr, size_t arr_len, const uint32_t index)
{
    const embedded_field elem = (index == 0 ? embedded_field(elems[index]) : embedded_field(tau[0]));
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		arr[tid] *= elem;
		tid += blockDim.x * gridDim.x;
	}
}

void mul_by_const(embedded_field* arr, size_t arr_len, const uint256_g& elem, const Geometry& geometry, 
    cudaStream_t& stream, uint32_t index)
{  
    mul_by_const_kernel<<<geometry.gridSize, geometry.blockSize, 0, stream>>>(arr, arr_len, index);
}

__global__ void mont_reduce_kernel(embedded_field* arr, size_t arr_len)
{
    const embedded_field elem = embedded_field(EMBEDDED_FIELD_R_inv);
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		arr[tid] *= elem;
		tid += blockDim.x * gridDim.x;
	}
}

void mont_reduce(embedded_field* arr, size_t arr_len, const Geometry& geometry, cudaStream_t& stream)
{  
    mont_reduce_kernel<<<geometry.gridSize, geometry.blockSize, 0, stream>>>(arr, arr_len);
}

DEVICE_FUNC embedded_field __inline__ get_gen_power(size_t index)
{
	embedded_field result(EMBEDDED_FIELD_R);

	//TODO: fix - index may be longer than 32 bits
    for (unsigned k = 0; k < 32; k++)
	{
		if (CHECK_BIT(index, k))
			result *= embedded_field(EMBEDDED_FIELD_MULT_GEN_ARR[k]);
	}
	return result;	
}

DEVICE_FUNC embedded_field __inline__ get_gen_inv_power(size_t index)
{
	embedded_field result(EMBEDDED_FIELD_R);

	for (unsigned k = 0; k < 32; k++)
	{
		if (CHECK_BIT(index, k))
			result *= embedded_field(EMBEDDED_FIELD_MULT_GEN_INV_ARR[k]);
	}
	return result;	
}

__global__ void distribute_powers_kernel(embedded_field* arr, size_t arr_len, bool is_inv)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	while (tid < arr_len)
	{		
        embedded_field elem = (is_inv ? get_gen_inv_power(tid) : get_gen_power(tid));
        arr[tid] *= elem;
        
		tid += blockDim.x * gridDim.x;
	}
}

void distribute_powers(embedded_field* arr, size_t arr_len, bool is_inv, const Geometry& geometry, cudaStream_t& stream)
{
    distribute_powers_kernel<<<geometry.gridSize, geometry.blockSize, 0, stream>>>(arr, arr_len, is_inv);
}


void FFT(embedded_field* arr, uint32_t arr_len, const Geometry& geometry, cudaStream_t& stream)
{
    fft_impl(arr, arr_len, false, geometry, stream);
}

void iFFT(embedded_field* arr, uint32_t arr_len, const uint256_g& inv, const Geometry& geometry, 
    cudaStream_t& stream, uint32_t index)
{
    fft_impl(arr, arr_len, true, geometry, stream);      
    mul_by_const(arr, arr_len, inv, geometry, stream, index);
}

void cosetFFT(embedded_field* arr, uint32_t arr_len, const Geometry& geometry, cudaStream_t& stream)
{
    distribute_powers(arr, arr_len, false, geometry, stream);
    fft_impl(arr, arr_len, false, geometry, stream);
}

void icosetFFT(embedded_field* arr, uint32_t arr_len, const uint256_g& inv, const Geometry& geometry, 
    cudaStream_t& stream, uint32_t index)
{
    fft_impl(arr, arr_len, true, geometry, stream);      
    mul_by_const(arr, arr_len, inv, geometry, stream, index);
    distribute_powers(arr, arr_len, true, geometry, stream);
}



//----------------------------------------------------------------------------------------------------------------------------------------------
//Groth 16 prover! (at least part of it)
//----------------------------------------------------------------------------------------------------------------------------------------------

//NB: the lengths of all these arrays should be equal!

void large_Pippenger_driver(affine_point*, uint256_g*, ec_point*, size_t);

struct Groth16_prover_data
{
    const uint8_t* a_arr;
    size_t a_len;

    const uint8_t* b_arr;
    size_t b_len;

    const uint8_t* c_arr;
    size_t c_len;

    const uint8_t* m_inv;
    const uint8_t* h_arr;
    const uint8_t* tau_inv;
    const uint8_t* check_arr;
};

size_t calc_domain_len(size_t len)
{
    size_t log_domain_len = BITS_PER_LIMB - __builtin_clz(len) - 1;
    size_t domain_len = (1 << log_domain_len);
    if (domain_len < len)
        domain_len *= 2;
    
    return domain_len;
}

affine_point Groth16_proof(const Groth16_prover_data* pr_data)
{
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

    if (!prop.deviceOverlap){
        exit( EXIT_FAILURE );
    }
    
    cudaStream_t stream1, stream2, stream3;
    HANDLE_ERROR(cudaStreamCreate(&stream1)); 
    HANDLE_ERROR(cudaStreamCreate(&stream2)); 
    HANDLE_ERROR(cudaStreamCreate(&stream3));

    const uint256_g* m_inv = (const uint256_g*)pr_data->m_inv;
    const uint256_g* tau_inv = (const uint256_g*)pr_data->tau_inv; 

    HANDLE_ERROR(cudaMemcpyToSymbol(elems, m_inv, sizeof(uint256_g), 0, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpyToSymbol(tau, tau_inv, sizeof(uint256_g), 0, cudaMemcpyHostToDevice));

    size_t a_domain_len = calc_domain_len(pr_data->a_len);
    size_t b_domain_len = calc_domain_len(pr_data->b_len);
    size_t c_domain_len = calc_domain_len(pr_data->c_len);

    assert(a_domain_len == b_domain_len);
    assert(b_domain_len == c_domain_len);

    size_t domain_len = a_domain_len;

    //lock memory and copy asynchroniously to device
    assert(mlock(pr_data->a_arr, pr_data->a_len * sizeof(embedded_field)) == 0);
    assert(mlock(pr_data->b_arr, pr_data->b_len * sizeof(embedded_field)) == 0);
    assert(mlock(pr_data->c_arr, pr_data->c_len * sizeof(embedded_field)) == 0);
    assert(mlock(pr_data->h_arr, domain_len * sizeof(affine_point)) == 0);

    embedded_field* dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr; 

    HANDLE_ERROR(cudaMalloc((void**)&dev_a, domain_len * sizeof(embedded_field)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b, domain_len * sizeof(affine_point)));
    HANDLE_ERROR(cudaMalloc((void**)&dev_c, domain_len * sizeof(embedded_field)));

    HANDLE_ERROR(cudaMemcpyAsync(dev_a, pr_data->a_arr, pr_data->a_len * sizeof(embedded_field), cudaMemcpyHostToDevice, stream1));
    HANDLE_ERROR(cudaMemcpyAsync(dev_b, pr_data->b_arr, pr_data->b_len * sizeof(embedded_field), cudaMemcpyHostToDevice, stream2));
    HANDLE_ERROR(cudaMemcpyAsync(dev_c, pr_data->c_arr, pr_data->c_len * sizeof(embedded_field), cudaMemcpyHostToDevice, stream3));
    	
    HANDLE_ERROR(cudaMemsetAsync(dev_a + pr_data->a_len, 0, (domain_len - pr_data->a_len) * sizeof(embedded_field), stream1));
    HANDLE_ERROR(cudaMemsetAsync(dev_b + pr_data->b_len, 0, (domain_len - pr_data->b_len) * sizeof(embedded_field), stream2));
    HANDLE_ERROR(cudaMemsetAsync(dev_c + pr_data->c_len, 0, (domain_len - pr_data->a_len) * sizeof(embedded_field), stream3));

    Geometry FFT_geometry = find_suitable_geometry(fft_iteration, 0, prop.multiProcessorCount);

    iFFT(dev_a, domain_len, *m_inv, FFT_geometry, stream1, 0);
    cosetFFT(dev_a, domain_len, FFT_geometry, stream1);

    iFFT(dev_b, domain_len, *m_inv, FFT_geometry, stream2, 0);
    cosetFFT(dev_b, domain_len, FFT_geometry, stream2);

    iFFT(dev_c, domain_len, *m_inv, FFT_geometry, stream3, 0);
    cosetFFT(dev_c, domain_len, FFT_geometry, stream3);

    HANDLE_ERROR( cudaStreamSynchronize( stream1 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream2 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream3 ) );

    fused_mul_sub(dev_a, dev_b, dev_c, domain_len, prop.multiProcessorCount);
    cudaDeviceSynchronize();
   
    mul_by_const(dev_a, domain_len, *tau_inv, FFT_geometry, stream1, 1);
    icosetFFT(dev_a, domain_len, *m_inv, FFT_geometry, stream1, 0);
    Geometry mont_reduce_geometry = find_suitable_geometry(mont_reduce_kernel, 0, prop.multiProcessorCount);
    mont_reduce(dev_a, domain_len - 1, mont_reduce_geometry, stream1);

    HANDLE_ERROR(cudaMemcpyAsync(dev_b, pr_data->h_arr, (domain_len - 1) * sizeof(affine_point), cudaMemcpyHostToDevice, stream2));

    HANDLE_ERROR( cudaStreamSynchronize( stream1 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream2 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream3 ) );

    HANDLE_ERROR( cudaStreamDestroy( stream1 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream2 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream3 ) );

    munlock(pr_data->a_arr, pr_data->a_len * sizeof(embedded_field));
    munlock(pr_data->b_arr, pr_data->b_len * sizeof(embedded_field));
    munlock(pr_data->c_arr, pr_data->c_len * sizeof(embedded_field));
    munlock(pr_data->h_arr, (domain_len - 1) * sizeof(affine_point));

    large_Pippenger_driver((affine_point*)dev_b, (uint256_g*)dev_a, (ec_point*)dev_c, domain_len - 1);

    affine_point res;
    HANDLE_ERROR(cudaMemcpy(&res, dev_c, sizeof(affine_point), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_a));
    HANDLE_ERROR(cudaFree(dev_b));
    HANDLE_ERROR(cudaFree(dev_c));

    return res;
}

extern "C"
{
    int EXPORT evaluate_h(size_t a_len, size_t b_len, size_t c_len, size_t h_len, const uint8_t* a_repr, const uint8_t* b_repr,
        const uint8_t* c_repr, const uint8_t* h_repr, const uint8_t* z_inv, const uint8_t* m_inv, uint8_t* result_ptr)
    {
        Groth16_prover_data pr_data;
        
        pr_data.a_arr = a_repr;
        pr_data.a_len = a_len;

        pr_data.b_arr = b_repr;
        pr_data.b_len = b_len;

        pr_data.c_arr = c_repr;
        pr_data.c_len = c_len;

        pr_data.m_inv = m_inv;
        pr_data.h_arr = h_repr;
        pr_data.tau_inv = z_inv;

        affine_point res = Groth16_proof(&pr_data);

        memcpy(result_ptr, &res, sizeof(affine_point));
        return 0;
    };

    //if flag in_mont_form = TRUE then tthe array of powers is in mont form and all the numbers should be converted to standard form
    //inside the CUDA kernel

    int EXPORT dense_multiexp(size_t len, const uint8_t* power_repr, const uint8_t* point_repr, bool repr_flag, uint8_t* result_ptr)
    {
        affine_point* dev_point_arr = nullptr;
        uint256_g* dev_power_arr  = nullptr;
        ec_point* dev_res = nullptr;

        HANDLE_ERROR(cudaMalloc((void**)&dev_point_arr, len * sizeof(affine_point)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_power_arr, len * sizeof(uint256_g)));
        HANDLE_ERROR(cudaMalloc((void**)&dev_res, sizeof(ec_point)));
  
        HANDLE_ERROR(cudaMemcpy(dev_point_arr, point_repr, len * sizeof(affine_point), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_power_arr, power_repr, len * sizeof(uint256_g), cudaMemcpyHostToDevice));

        if (repr_flag)
        {
            cudaDeviceProp prop;
            HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

            Geometry mont_reduce_geometry = find_suitable_geometry(mont_reduce_kernel, 0, prop.multiProcessorCount);
            cudaStream_t stream = 0;
            mont_reduce((embedded_field*)dev_power_arr, len, mont_reduce_geometry, stream);
        }

        large_Pippenger_driver(dev_point_arr, dev_power_arr, dev_res, len);

        affine_point res;
        HANDLE_ERROR(cudaMemcpy(&res, dev_res, sizeof(affine_point), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_point_arr));
        HANDLE_ERROR(cudaFree(dev_power_arr));
        HANDLE_ERROR(cudaFree(dev_res));

        memcpy(result_ptr, &res, sizeof(affine_point));
        return 0;
    };


}


int main(int argc, char* argv[])
{
    return 0;
}
    

	