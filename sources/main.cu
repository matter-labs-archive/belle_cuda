#include "cuda_structs.h"

#include <chrono>
#include <stdlib.h>

#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <vector>

#include <stdio.h>
#include <time.h>

//extern function we are going to use
bool CUDA_init();
ec_point get_random_point_host();

template<typename T>
T get_random_elem();

template<>
uint128_g get_random_elem<uint128_g>()
{
    uint128_g res;
    for (uint32_t i =0; i < HALF_N; i++)
        res.n[i] = rand();
    return res;
}

template<>
uint256_g get_random_elem<uint256_g>()
{
    uint256_g res;
    for (uint32_t i =0; i < N; i++)
        res.n[i] = rand();
    res.n[N - 1] &= 0x1fffffff;
    return res;
}

template<>
ec_point get_random_elem<ec_point>()
{
    return get_random_point_host();
}

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

template<typename Atype, typename Btype, typename Ctype>
using kernel_func_ptr = void (*)(Atype*, Btype*, Ctype*, size_t);

template<typename Atype, typename Btype, typename Ctype>
using kernel_func_vec_t = std::vector<std::pair<const char*, kernel_func_ptr<Atype, Btype, Ctype>>>;

template<typename Atype, typename Btype, typename Ctype>
void gpu_benchmark(kernel_func_vec_t<Atype, Btype, Ctype> func_vec, size_t bench_len)
{
    Atype* A_host_arr = nullptr;
    Btype* B_host_arr = nullptr;
    Ctype* C_host_arr = nullptr;

    Atype* A_dev_arr = nullptr;
    Btype* B_dev_arr = nullptr;
    Ctype* C_dev_arr = nullptr;

    auto num_of_kernels = func_vec.size();
    std::chrono::high_resolution_clock::time_point start, end;   
    std::int64_t duration;

    cudaError_t cudaStatus;

    //fill in A array
    A_host_arr = (Atype*)malloc(bench_len * sizeof(Atype));
    for (size_t i = 0; i < bench_len; i++)
    {
        A_host_arr[i] = get_random_elem<Atype>();
    }

#ifdef PRINT_BENCHES
    std::cout << "A array:" << std::endl;
    for (size_t i = 0; i < bench_len; i++)
    {
        std::cout << A_host_arr[i] << std::endl;
    }
    std::cout << std::endl;
#endif

    //fill in B array
    B_host_arr = (Btype*)malloc(bench_len * sizeof(Btype));
    for (size_t i = 0; i < bench_len; i++)
    {
        B_host_arr[i] = get_random_elem<Btype>();
    }

#ifdef PRINT_BENCHES
    std::cout << "B array:" << std::endl;
    for (size_t i = 0; i < bench_len; i++)
    {
        std::cout << B_host_arr[i] << std::endl;
    }
    std::cout << std::endl;
#endif

    //allocate C array
    C_host_arr = (Ctype*)malloc(bench_len * sizeof(Ctype));
 
    cudaStatus = cudaMalloc(&A_dev_arr, bench_len * sizeof(Atype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (A_dev_arr) failed!\n");
        fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc(&B_dev_arr, bench_len * sizeof(Btype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (B_dev_arr) failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc(&C_dev_arr, bench_len * sizeof(Ctype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (C_dev_arr) failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(A_dev_arr, A_host_arr, bench_len * sizeof(Atype), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (A_arrs) failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(B_dev_arr, B_host_arr, bench_len * sizeof(Btype), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (B_arrs) failed!");
        goto Error;
    }

     start = std::chrono::high_resolution_clock::now();

    //run_kernels!
    //---------------------------------------------------------------------------------------------------------------------------------
    for(size_t i = 0; i < num_of_kernels; i++)
    {
        auto f = func_vec[i].second;
        auto message = func_vec[i].first;

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
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
        std::cout << "ns total GPU on func " << message <<":   "  << std::dec << duration  << "ns." << std::endl;

        cudaStatus = cudaMemcpy(C_host_arr, C_dev_arr, bench_len * sizeof(Ctype), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy (C_arrs) failed!");
            goto Error;
        }

#ifdef PRINT_BENCHES
        std::cout << "C array:" << std::endl;
        for (size_t i = 0; i < bench_len; i++)
        {
            std::cout << C_host_arr[i] << std::endl;
        }
        std::cout << std::endl;
#endif
    }

Error:
    cudaFree(A_dev_arr);
    cudaFree(B_dev_arr);
    cudaFree(C_dev_arr);

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

mul_func_vec_t mul_bench = {
    {"naive approach", mul_uint256_to_512_naive_driver},
	{"asm", mul_uint256_to_512_asm_driver},
	{"asm with register alloc", mul_uint256_to_512_asm_with_allocation_driver},
	{"asm with longregs", mul_uint256_to_512_asm_longregs_driver},
    {"Karatsuba", mul_uint256_to_512_Karatsuba_driver}
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

ecc_point_exp_func_vec_t exp_curve_point_bench = {
    {"double and add in projective coordinates", ECC_double_and_add_exp_PROJ_driver},
    {"exp via ternary expansion in projective coordinates", ECC_ternary_expansion_exp_PROJ_driver},
    {"double and add in Jacobian coordinates", ECC_double_and_add_exp_JAC_driver},
    {"exp via ternary expansion in Jacobian coordinates", ECC_ternary_expansion_exp_JAC_driver}
};

//-------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------


size_t bench_len = 0x3;

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
	
	// std::cout << "addition benchmark: " << std::endl;
	// gpu_benchmark(addition_bench, bench_len);

	// std::cout << "substraction benchmark: " << std::endl;
	// gpu_benchmark(substraction_bench, bench_len);

	// std::cout << "multiplication benchmark: " << std::endl;
	// gpu_benchmark(mul_bench, bench_len);

	// std::cout << "square benchmark: " << std::endl;
	// gpu_benchmark(square_bench, bench_len);

	// std::cout << "montgomery multiplication benchmark: " << std::endl;
	// gpu_benchmark(mont_mul_bench, bench_len);

    // std::cout << "ECC add-sub benchmark: " << std::endl;
    // gpu_benchmark(add_sub_curve_points_bench, bench_len);

    // std::cout << "ECC double benchmark: " << std::endl;
    // gpu_benchmark(double_curve_point_bench, bench_len);

    std::cout << "ECC exponentiation benchmark: " << std::endl;
    gpu_benchmark(exp_curve_point_bench, bench_len);

    return 0;
}



