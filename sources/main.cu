#include "cuda_structs.h"

#include <chrono>
#include <stdlib.h>

#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <stdio.h>
#include <time.h>

#include <iostream>
#include <fstream> 
#include <iomanip>

std::ostream& operator<<(std::ostream& os, const uint256_g num)
{
    os << "0x";
    for (int i = 7; i >= 0; i--)
    {
        os << std::setfill('0') << std::hex << std::setw(8) << num.n[i];
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const uint512_g num)
{
    os << "0x";
    for (int i = 15; i >= 0; i--)
    {
        os << std::setfill('0') << std::hex << std::setw(8) << num.n[i];
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const ec_point& pt)
{
    os << "x = " << pt.x << std::endl;
    os << "y = " << pt.y << std::endl;
    os << "z = " << pt.z << std::endl;
    return os;
}

std::ostream& operator<<(std::ostream& os, const affine_point& pt)
{
    os << "x = " << pt.x << std::endl;
    os << "y = " << pt.y << std::endl;
    return os;
}


template<typename Atype, typename Btype, typename Ctype>
using kernel_func_ptr = void (*)(Atype*, Btype*, Ctype*, size_t);

template<typename Atype, typename Btype, typename Ctype>
using kernel_func_vec_t = std::vector<std::pair<const char*, kernel_func_ptr<Atype, Btype, Ctype>>>;

template<typename Atype, typename Btype, typename Ctype>
void gpu_benchmark(kernel_func_vec_t<Atype, Btype, Ctype> func_vec, size_t bench_len, const char* output_file = nullptr, 
    bool scalar_return = false)
{
    Atype* A_host_arr = nullptr;
    Btype* B_host_arr = nullptr;
    Ctype* C_host_arr = nullptr;

    Atype* A_dev_arr = nullptr;
    Btype* B_dev_arr = nullptr;
    Ctype* C_dev_arr = nullptr;

    curandState *devStates = nullptr;

    std::ofstream fstream;
    if (output_file)
        fstream.open(output_file);
    std::ostream& stream = (output_file ? fstream : std::cout);

    int blockSize;
  	int minGridSize;
    int realGridSize;
	int optimalGridSize;

    auto num_of_kernels = func_vec.size();
    std::chrono::high_resolution_clock::time_point start, end;   
    std::int64_t duration;

    cudaError_t cudaStatus;

    //allocate device arrays

    cudaStatus = cudaMalloc(&A_dev_arr, bench_len * sizeof(Atype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (A_dev_arr) failed!\n");
        //fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&B_dev_arr, bench_len * sizeof(Btype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (B_dev_arr) failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc(&C_dev_arr, 1000 * sizeof(Ctype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (C_dev_arr) failed!\n");
        goto Error;
    }

    //we generate all random variables on the device side - below is the PRNG initialization

  	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, gen_random_array_kernel<Atype>, 0, 0);
  	realGridSize = (bench_len + blockSize - 1) / blockSize;
	optimalGridSize = min(minGridSize, realGridSize);


    cudaStatus = cudaMalloc((void **)&devStates, optimalGridSize * blockSize * sizeof(curandState));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (devStates) failed!\n");
        goto Error;
    }

    //generate ranodm elements in both input arrays

    gen_random_array_kernel<<<optimalGridSize, blockSize>>>(A_dev_arr, bench_len, devStates, rand());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "random elements generator kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    gen_random_array_kernel<<<optimalGridSize, blockSize>>>(B_dev_arr, bench_len, devStates, rand());

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "random elements generator kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


#ifdef PRINT_BENCHES

    stream << "Bench len = " << bench_len << std::endl;

    //copy generated arrays from device to host and print them!

    A_host_arr = (Atype*)malloc(bench_len * sizeof(Atype));
    B_host_arr = (Btype*)malloc(bench_len * sizeof(Btype));
    if (scalar_return)
         C_host_arr = (Ctype*)malloc(sizeof(Ctype));
    else
        C_host_arr = (Ctype*)malloc(1000 * sizeof(Ctype));

    cudaStatus = cudaMemcpy(A_host_arr, A_dev_arr, bench_len * sizeof(Atype), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (A_arrs) failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(B_host_arr, B_dev_arr, bench_len * sizeof(Btype), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (B_arrs) failed!\n");
        goto Error;
    }

    stream << "A array:" << std::endl;
    for (size_t i = 0; i < bench_len; i++)
    {
        stream << A_host_arr[i] << std::endl;
    }
    stream << std::endl;

    stream << "B array:" << std::endl;
    for (size_t i = 0; i < bench_len; i++)
    {
        stream << B_host_arr[i] << std::endl;
    }
    stream << std::endl;

#endif

    //run_kernels!
    //---------------------------------------------------------------------------------------------------------------------------------
    for(size_t i = 0; i < num_of_kernels; i++)
    {
        auto f = func_vec[i].second;
        auto message = func_vec[i].first;

        std::cout << "Launching kernel: "  << message << std::endl;

        start = std::chrono::high_resolution_clock::now();
        f(A_dev_arr, B_dev_arr, C_dev_arr, bench_len);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel %s!\n", cudaStatus, message);
            goto Error;
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        std::cout << "ns total GPU: "  << std::dec << duration  << "ns." << std::endl << std::endl << std::endl;

#ifdef PRINT_BENCHES

        if (scalar_return)
        {
            cudaStatus = cudaMemcpy(C_host_arr, C_dev_arr, sizeof(Ctype), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy (C_elem) failed!\n");
                goto Error;
            }

            stream << "C elem:" << std::endl;
            stream << C_host_arr[0] << std::endl;
            stream << std::endl;
        }
        else
        {
            cudaStatus = cudaMemcpy(C_host_arr, C_dev_arr, 1000 * sizeof(Ctype), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "cudaMemcpy (C_arrs) failed!\n");
                goto Error;
            }

            stream << "C array:" << std::endl;
            for (size_t i = 0; i < 512; i++)
            {
                stream << C_host_arr[i] << std::endl;
            }
            stream << std::endl;
        }
        
#endif
    }

Error:
    cudaFree(A_dev_arr);
    cudaFree(B_dev_arr);
    cudaFree(C_dev_arr);

    cudaFree(devStates);

    free(A_host_arr);
    free(B_host_arr);
    free(C_host_arr);
}


using mul_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint512_g>;
using general_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint256_g>;

void add_uint256_naive_driver(uint256_g*, uint256_g*, uint256_g*, size_t);
void add_uint256_asm_driver(uint256_g*, uint256_g*, uint256_g*, size_t);

general_func_vec_t addition_bench = {
    {"naive approach", add_uint256_naive_driver},
	{"asm", add_uint256_asm_driver}
};

void sub_uint256_naive_driver(uint256_g*, uint256_g*, uint256_g*, size_t);
void sub_uint256_asm_driver(uint256_g*, uint256_g*, uint256_g*, size_t);

general_func_vec_t substraction_bench = {
    {"naive approach", sub_uint256_naive_driver},
	{"asm", sub_uint256_asm_driver}
};

void mul_uint256_to_512_naive_driver(uint256_g*, uint256_g*, uint512_g*, size_t);
void mul_uint256_to_512_asm_driver(uint256_g*, uint256_g*, uint512_g*, size_t);
void mul_uint256_to_512_asm_with_allocation_driver(uint256_g*, uint256_g*, uint512_g*, size_t);
void mul_uint256_to_512_asm_longregs_driver(uint256_g*, uint256_g*, uint512_g*, size_t);
void mul_uint256_to_512_Karatsuba_driver(uint256_g*, uint256_g*, uint512_g*, size_t);
void mul_uint256_to_512_asm_with_shuffle_driver(uint256_g*, uint256_g*, uint512_g*, size_t);

mul_func_vec_t mul_bench = {
    {"naive approach", mul_uint256_to_512_naive_driver},
	{"asm", mul_uint256_to_512_asm_driver},
	{"asm with register alloc", mul_uint256_to_512_asm_with_allocation_driver},
	{"asm with longregs", mul_uint256_to_512_asm_longregs_driver},
    {"Karatsuba", mul_uint256_to_512_Karatsuba_driver},
    {"asm with shuffles", mul_uint256_to_512_asm_with_shuffle_driver}
};

void square_uint256_to_512_naive_driver(uint256_g*, uint256_g*, uint512_g*, size_t);
void square_uint256_to_512_asm_driver(uint256_g*, uint256_g*, uint512_g*, size_t);

mul_func_vec_t square_bench = {
    {"naive approach", square_uint256_to_512_naive_driver},
	{"asm", square_uint256_to_512_asm_driver},
};

void mont_mul_256_naive_SOS_driver(uint256_g*, uint256_g*, uint256_g*, size_t);
void mont_mul_256_asm_SOS_driver(uint256_g*, uint256_g*, uint256_g*, size_t);
void mont_mul_256_naive_CIOS_driver(uint256_g*, uint256_g*, uint256_g*, size_t);
void mont_mul_256_asm_CIOS_driver(uint256_g*, uint256_g*, uint256_g*, size_t);

general_func_vec_t mont_mul_bench = {
    {"naive SOS", mont_mul_256_naive_SOS_driver},
	{"asm SOS", mont_mul_256_asm_SOS_driver},
	{"naive CIOS", mont_mul_256_naive_CIOS_driver},
	{"asm CIOS", mont_mul_256_asm_CIOS_driver}
};


using mul_inv_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint256_g>;

void FIELD_MUL_INV_driver(uint256_g*, uint256_g*, uint256_g*, size_t);

mul_inv_func_vec_t mul_inv_bench = {
    {"mul inversion", FIELD_MUL_INV_driver}
};

using ecc_general_func_vec_t = kernel_func_vec_t<ec_point, ec_point, ec_point>;
using ecc_point_exp_func_vec_t = kernel_func_vec_t<ec_point, uint256_g, ec_point>;

void ECC_ADD_PROJ_driver(ec_point*, ec_point*, ec_point*, size_t);
void ECC_ADD_JAC_driver(ec_point*, ec_point*, ec_point*, size_t);
void ECC_SUB_PROJ_driver(ec_point*, ec_point*, ec_point*, size_t);
void ECC_SUB_JAC_driver(ec_point*, ec_point*, ec_point*, size_t);

ecc_general_func_vec_t add_sub_curve_points_bench = {
    {"addition in projective coordinates", ECC_ADD_PROJ_driver},
	{"addition in Jacobian coordinates", ECC_ADD_JAC_driver},
	{"substraction in projective coordinates", ECC_SUB_PROJ_driver},
	{"substraction in Jacobian coordinates", ECC_SUB_JAC_driver}
};

void ECC_DOUBLE_PROJ_driver(ec_point*, ec_point*, ec_point*, size_t);
void ECC_DOUBLE_JAC_driver(ec_point*, ec_point*, ec_point*, size_t);

ecc_general_func_vec_t double_curve_point_bench = {
    {"doubling in projective coordinates", ECC_DOUBLE_PROJ_driver},
    {"doubling in Jacobian coordinates", ECC_DOUBLE_JAC_driver}
};

void ECC_double_and_add_exp_PROJ_driver(ec_point*, uint256_g*, ec_point*, size_t);
void ECC_ternary_expansion_exp_PROJ_driver(ec_point*, uint256_g*, ec_point*, size_t);
void ECC_double_and_add_exp_JAC_driver(ec_point*, uint256_g*, ec_point*, size_t);
void ECC_ternary_expansion_exp_JAC_driver(ec_point*, uint256_g*, ec_point*, size_t);
void ECC_wNAF_exp_PROJ_driver(ec_point*, uint256_g*, ec_point*, size_t);
void ECC_wNAF_exp_JAC_driver(ec_point*, uint256_g*, ec_point*, size_t);

ecc_point_exp_func_vec_t exp_curve_point_bench = {
    {"double and add in projective coordinates", ECC_double_and_add_exp_PROJ_driver},
    {"exp via ternary expansion in projective coordinates", ECC_ternary_expansion_exp_PROJ_driver},
    {"wNaf exp in projective coordinates", ECC_wNAF_exp_PROJ_driver},
    {"double and add in Jacobian coordinates", ECC_double_and_add_exp_JAC_driver},
    {"exp via ternary expansion in Jacobian coordinates", ECC_ternary_expansion_exp_JAC_driver},
    {"wNaf exp in Jacobian coordinates", ECC_wNAF_exp_JAC_driver}
};


using ecc_point_affine_exp_func_vec_t = kernel_func_vec_t<affine_point, uint256_g, ec_point>;

void ECC_double_and_add_affine_exp_PROJ_driver(affine_point*, uint256_g*, ec_point*, size_t);
void ECC_double_and_add_affine_exp_JAC_driver(affine_point*, uint256_g*, ec_point*, size_t);

ecc_point_affine_exp_func_vec_t affine_exp_curve_point_bench = {
    {"double and add in projective coordinates", ECC_double_and_add_affine_exp_PROJ_driver},
    {"double and add in Jacobian coordinates", ECC_double_and_add_affine_exp_JAC_driver}
};


using ecc_multiexp_func_vec_t = kernel_func_vec_t<affine_point, uint256_g, ec_point>;

void naive_multiexp_kernel_warp_level_atomics_driver(affine_point*, uint256_g*, ec_point*, size_t);
void naive_multiexp_kernel_block_level_atomics_driver(affine_point*, uint256_g*, ec_point*, size_t);
void naive_multiexp_kernel_block_level_recursion_driver(affine_point*, uint256_g*, ec_point*, size_t);
void small_Pippenger_driver(affine_point*, uint256_g*, ec_point*, size_t);
void large_Pippenger_driver(affine_point*, uint256_g*, ec_point*, size_t);

ecc_multiexp_func_vec_t multiexp_curve_point_bench = {
    {"naive warp level approach with atomics", naive_multiexp_kernel_warp_level_atomics_driver},
    //{"naive block level approach with atomics", naive_multiexp_kernel_block_level_atomics_driver},
    //{"naive block level approach with recursion", naive_multiexp_kernel_block_level_recursion_driver},
    {"Pippenger: 2**8 elems per bin", small_Pippenger_driver},
    {"Pippenger: 2**16 elems per bin", large_Pippenger_driver},
};

//-------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------


size_t bench_len = 1000000;
//size_t bench_len = 10;

const char* OUTPUT_FILE = "benches.txt";

int main(int argc, char* argv[])
{
    long ltime = time (NULL);
    unsigned int stime = (unsigned int) ltime/2;
    srand(stime);

    bool result = CUDA_init();

	if (!result)
	{
		printf("error");
		return -1;
	}

    std::cout << "Benchmark length: " << bench_len << std::endl << std::endl;
	
	// std::cout << "addition benchmark: " << std::endl << std::endl;
	// gpu_benchmark(addition_bench, bench_len);

	// std::cout << "substraction benchmark: " << std::endl << std::endl;
	// gpu_benchmark(substraction_bench, bench_len);

	// std::cout << "multiplication benchmark: " << std::endl << std::endl;
	// gpu_benchmark(mul_bench, bench_len);

	// std::cout << "square benchmark: " << std::endl << std::endl;
	// gpu_benchmark(square_bench, bench_len);

    // std::cout << "field inversion benchmark: " << std::endl << std::endl;
	// gpu_benchmark(mul_inv_bench, bench_len);

	// std::cout << "montgomery multiplication benchmark: " << std::endl << std::endl;
	// gpu_benchmark(mont_mul_bench, bench_len);

    // std::cout << "ECC add-sub benchmark: " << std::endl << std::endl;
    // gpu_benchmark(add_sub_curve_points_bench, bench_len);

    // std::cout << "ECC double benchmark: " << std::endl << std::endl;
    // gpu_benchmark(double_curve_point_bench, bench_len);

    // std::cout << "ECC exponentiation benchmark: " << std::endl << std::endl;
    // gpu_benchmark(exp_curve_point_bench, bench_len);

    // std::cout << "ECC affine exponentiation benchmark: " << std::endl << std::endl;
    // gpu_benchmark(affine_exp_curve_point_bench, bench_len, OUTPUT_FILE);

    std::cout << "ECC multi-exponentiation benchmark: " << std::endl << std::endl;
    gpu_benchmark(multiexp_curve_point_bench, bench_len, nullptr, false);

    return 0;
}





