#include "cuda_structs.h"
#include "cuda_export_headers.h"

struct geometry_local
{
    int gridSize;
    int blockSize;
};

template<typename T>
geometry_local find_suitable_geometry_local(T func, uint shared_memory_used, uint32_t smCount)
{
    int gridSize;
    int blockSize;
    int maxActiveBlocks;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, shared_memory_used, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, shared_memory_used);
    gridSize = maxActiveBlocks * smCount;

    return geometry_local{gridSize, blockSize};
}


__global__ void field_add_kernel(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = a_arr[tid] + b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void field_sub_kernel(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = a_arr[tid] - b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void field_mul_kernel(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, size_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = a_arr[tid] * b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

using field_kernel_t = __global__ void(const embedded_field*, const embedded_field*, embedded_field*, size_t); 


void field_func_invoke(const embedded_field* a_host_arr, const embedded_field* b_host_arr, embedded_field* c_host_arr, uint32_t arr_len,
    field_kernel_t func)
{
	cudaDeviceProp prop;
  	cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

    geometry_local geometry = find_suitable_geometry_local(func, 0, smCount);

    embedded_field* a_dev_arr = nullptr;
    embedded_field* b_dev_arr = nullptr;
    embedded_field* c_dev_arr = nullptr;

    cudaMalloc((void **)&a_dev_arr, arr_len * sizeof(embedded_field));
    cudaMalloc((void **)&b_dev_arr, arr_len * sizeof(embedded_field));
    cudaMalloc((void **)&c_dev_arr, arr_len * sizeof(embedded_field));

    cudaMemcpy(a_dev_arr, a_host_arr, arr_len * sizeof(embedded_field), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev_arr, b_host_arr, arr_len * sizeof(embedded_field), cudaMemcpyHostToDevice);

    (*func)<<<geometry.gridSize, geometry.blockSize, 0>>>(a_dev_arr, b_dev_arr, c_dev_arr, arr_len);

    cudaMemcpy(c_host_arr, c_dev_arr, arr_len * sizeof(embedded_field), cudaMemcpyDeviceToHost);
    
    cudaFree(a_dev_arr);
    cudaFree(b_dev_arr);
    cudaFree(c_dev_arr);
}

void field_add(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, uint32_t arr_len)
{
    field_func_invoke(a_arr, b_arr, c_arr, arr_len, field_add_kernel);
}

void field_sub(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, uint32_t arr_len)
{
    field_func_invoke(a_arr, b_arr, c_arr, arr_len, field_sub_kernel);
}

void field_mul(const embedded_field* a_arr, const embedded_field* b_arr, embedded_field* c_arr, uint32_t arr_len)
{
    field_func_invoke(a_arr, b_arr, c_arr, arr_len, field_mul_kernel);
}


__global__ void ec_add_kernel(const ec_point* a_arr, const ec_point* b_arr, ec_point* c_arr, uint32_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = ECC_ADD(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void ec_sub_kernel(const ec_point* a_arr, const ec_point* b_arr, ec_point* c_arr, uint32_t arr_len)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		c_arr[tid] = ECC_SUB(a_arr[tid], b_arr[tid]);
		tid += blockDim.x * gridDim.x;
	}
}

using ec_kernel_t = __global__ void(const ec_point*, const ec_point*, ec_point*, size_t); 

void ec_func_invoke(const ec_point* a_host_arr, const ec_point* b_host_arr, ec_point* c_host_arr, uint32_t arr_len,
    ec_kernel_t func)
{
	cudaDeviceProp prop;
  	cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

    geometry_local geometry = find_suitable_geometry_local(func, 0, smCount);

    ec_point* a_dev_arr = nullptr;
    ec_point* b_dev_arr = nullptr;
    ec_point* c_dev_arr = nullptr;

    cudaMalloc((void **)&a_dev_arr, arr_len * sizeof(ec_point));
    cudaMalloc((void **)&b_dev_arr, arr_len * sizeof(ec_point));
    cudaMalloc((void **)&c_dev_arr, arr_len * sizeof(ec_point));

    cudaMemcpy(a_dev_arr, a_host_arr, arr_len * sizeof(ec_point), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev_arr, b_host_arr, arr_len * sizeof(ec_point), cudaMemcpyHostToDevice);

    (*func)<<<geometry.gridSize, geometry.blockSize, 0>>>(a_dev_arr, b_dev_arr, c_dev_arr, arr_len);

    cudaMemcpy(c_host_arr, c_dev_arr, arr_len * sizeof(ec_point), cudaMemcpyDeviceToHost);
    
    cudaFree(a_dev_arr);
    cudaFree(b_dev_arr);
    cudaFree(c_dev_arr);
}

void ec_point_add(ec_point* a_arr, ec_point* b_arr, ec_point* c_arr, uint32_t arr_len)
{
    ec_func_invoke(a_arr, b_arr, c_arr, arr_len, ec_add_kernel);
}

void ec_point_sub(ec_point* a_arr, ec_point* b_arr, ec_point* c_arr, uint32_t arr_len)
{
    ec_func_invoke(a_arr, b_arr, c_arr, arr_len, ec_sub_kernel);
}

//-----------------------------------------------------------------------------------------------------------------------------------------------
//Multiexponentiation (based on Pippenger realization)
//-----------------------------------------------------------------------------------------------------------------------------------------------

void large_Pippenger_driver(affine_point*, uint256_g*, ec_point*, size_t);

ec_point ec_multiexp(affine_point* points, uint256_g* powers, uint32_t arr_len)
{
    
    affine_point* dev_points = nullptr;
    uint256_g* dev_powers = nullptr;
    ec_point* dev_res = nullptr;

    ec_point res;

    cudaMalloc((void **)&dev_points, arr_len * sizeof(affine_point));
    cudaMalloc((void **)&dev_powers, arr_len * sizeof(uint256_g));
    cudaMalloc((void **)&dev_res, ec_point);

    cudaMemcpy(dev_points, points, arr_len * sizeof(affine_point), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_powers, powers, arr_len * sizeof(uint256_g), cudaMemcpyHostToDevice);

    large_Pippenger_driver(dev_points, dev_powers, dev_res, arr_len);

    cudaMemcpy(&res, dev_res, sizeof(ec_point), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_points);
    cudaFree(dev_powers);
    cudaFree(dev_res);
    
    return res;
}

//-----------------------------------------------------------------------------------------------------------------------------------------------
//FFT routines
//-----------------------------------------------------------------------------------------------------------------------------------------------

void naive_fft_driver(embedded_field*, embedded_field*, uint32_t, bool);

void mult_by_const(embedded_field* arr, __constant__ embedded_field& elem)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{
		arr[tid] *= elem;
		tid += blockDim.x * gridDim.x;
	}
}

void FFT_invoke(embedded_field* input_arr, embedded_field* output_arr, uint32_t arr_len, bool is_inverse, embedded_field* inversed = nullptr)
{
    embedded_field* dev_input_arr = nullptr;
    embedded_field* dev_output_arr = nullptr;

    cudaMalloc((void **)&dev_input_arr, arr_len * sizeof(embedded_field));
    cudaMalloc((void **)&dev_output_arr, arr_len * sizeof(embedded_field));

    cudaMemcpy(dev_input_arr, input_arr, arr_len * sizeof(embedded_field), cudaMemcpyHostToDevice);
    naive_fft_driver(dev_input_arr, dev_output_arr, arr_len, is_inverse);
   
    if (is_inverse)
    {
        __constant__ embedded_field dev_temp;
        cudaMemcpyToSymbol(dev_temp, inversed, sizeof(embedded_field));
        
        mult_by_const(output_arr, dev_temp);
    }
   
    cudaMemcpy(output_arr, dev_output_arr, arr_len * sizeof(embedded_field), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_input_arr);
    cudaFree(dev_output_arr);
}

void EXPORT FFT(embedded_field* input_arr, embedded_field* output_arr, uint32_t arr_len)
{
    FFT_invoke(input_arr, output_arr, arr_len, false);
}

void EXPORT iFFT(embedded_field* input_arr, embedded_field* output_arr, uint32_t arr_len, const embedded_field& n_inv)
{
    FFT_invoke(input_arr, output_arr, arr_len, true, &n_inv);
}

//------------------------------------------------------------------------------------------------------------------------------------------------
//polynomial arithmetic
//------------------------------------------------------------------------------------------------------------------------------------------------

// polynomial _polynomial_multiplication_on_fft(const polynomial&, const polynomial&);

// polynomial EXPORT poly_add(const& polynomial, const& polynomial);
// polynomial EXPORT poly_sub(const& polynomial, const& polynomial);
// polynomial EXPORT poly_mul(const& polynomial, const& polynomial);

