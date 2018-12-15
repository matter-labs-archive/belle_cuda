#ifndef BENCHMARK_CUH
#define BENCHMARK_CUH

#include "mul_256_to_512.cuh"

#include <chrono>
#include <vector>
#include <stdlib.h>
#include <iostream>


template<typename T>
T get_random_elem();

template<>
uint128_g get_random_elem<uint128_g>()
{
    uint128_g res;
    for (uint32_t i =0; i < 4; i++)
        res.n[i] = rand();
    return res;
}

template<>
uint256_g get_random_elem<uint256_g>()
{
    uint256_g res;
    for (uint32_t i =0; i < 8; i++)
        res.n[i] = rand();
    return res;
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

    //fill in B array
    B_host_arr = (Btype*)malloc(bench_len * sizeof(Btype));
    for (size_t i = 0; i < bench_len; i++)
    {
        B_host_arr[i] = get_random_elem<Btype>();
    }

    //allocate C array
    C_host_arr = (Ctype*)malloc(bench_len * sizeof(Ctype));

    cudaStatus = cudaMalloc(&A_dev_arr, bench_len * sizeof(Atype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (A_dev_arr) failed!");
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
        std::cout << "ns total GPU on func " << message <<":   "  << duration  << "ns." << std::endl;
    }

    cudaStatus = cudaMemcpy(C_host_arr, C_dev_arr, bench_len * sizeof(Ctype), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (C_arrs) failed!");
        goto Error;
    }

Error:
    cudaFree(A_dev_arr);
    cudaFree(B_dev_arr);
    cudaFree(C_dev_arr);

    free(A_host_arr);
    free(B_host_arr);
    free(C_host_arr);
}


#endif