#include "cuda_structs.h"

#include <vector>
#include <string>
#include <string_view>
#include <stdexcept>

#include <boost/convert.hpp>
#include <boost/convert/stream.hpp>
#include <boost/optional>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

static constexpr size_t BYTES_PER_LIMB = 4;

template<typename T>
T unpack_from_string(const std::string&)
{
    static constexpr size_t chars_per_limb = 2 * BYTES_PER_LIMB;

	const size_t str_len = str.size();

    size_t LIMB_COUNT = (std::is_same<T, uint256_g>::value ? 8 : 16);

	assert(str_len <= 2 * bytes_per_limb * LIMB_COUNT);

    T res;
    for (size_t i = 0; i < LIMB_COUNT; i++)
        res.n[i] = 0;

	if (str_len == 0)
		return;

    boost::cnv::cstream ccnv;
    ccnv(std::hex)(std::skipws);

    size_t i = str_len;
    limb_index_t limb_index = 0;
    while (i > 0)
    {
        size_t j = (2 * bytes_per_limb > i ? 0 : i - 2 * bytes_per_limb);
        std::string_view str_view(str.c_str() + j, i - j);
        i -= (i > 2 * bytes_per_limb ? 2 * bytes_per_limb : i);
        auto opt_val = boost::convert<uint32_t>(str_view, ccnv);
        if (opt_val)
            res.n[limb_index++] = opt_val.get();
        else
            throw std::runtime_error("Incorrect conversion");
    }

    return res;
}

template<>
ec_point unpack_from_string<ec_point>(const std::string& str)
{
    std::vector<std::string> strings;
    std::istringstream str_stream(str);
    std::string s;    
    while (getline(str_stream, s, ','))
        strings.push_back(s);

    assert(string.size() == 3);

    ec_point res;
    res.x = unpack_from_string<uint256_g>(strings[0]);
    res.y = unpack_from_string<uint256_g>(strings[1]);
    res.z = unpack_from_string<uint256_g>(strings[2]);

    return res;
}

template<typename Atype, typename Btype, typename Ctype>
using kernel_func_ptr = void (*)(Atype*, Btype*, Ctype*, size_t);

template<typename Atype, typename Btype, typename Ctype>
using kernel_func_vec_t = std::vector<std::pair<const char*, kernel_func_ptr<Atype, Btype, Ctype>>>;

template<typename Atype, typename Btype, typename Ctype>
bool test_framework(kernel_func_vec_t<Atype, Btype, Ctype> func_vec, Atype* A_host_arr, Btype* B_host_arr, Ctype* C_host_arr 
    std::vector<Ctype*>& results, const std::vector<std::string>& data, size_t bench_len)
{
    Atype* A_dev_arr = nullptr;
    Btype* B_dev_arr = nullptr;
    Ctype* C_dev_arr = nullptr;

    bool is_successful = true;

    auto num_of_kernels = func_vec.size(); 
    cudaError_t cudaStatus;

    //fill in host arrays
    size_t data_index = 0;
    for (size_t i = 0; i < bench_len; i++)
    {
        A_host_arr[i] = unpack_from_string<Atype>(data[data_index++]);
        B_host_arr[i] = unpack_from_string<Btype>(data[data_index++]);
        C_host_arr[i] = unpack_from_string<Btype>(data[data_index++]);
    }
   
    cudaStatus = cudaMalloc(&A_dev_arr, bench_len * sizeof(Atype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (A_dev_arr) failed!\n");
        is_successful = false;
        goto Error;
    }

    cudaStatus = cudaMalloc(&B_dev_arr, bench_len * sizeof(Btype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (B_dev_arr) failed!\n");
        is_successful = false;
        goto Error;
    }

    cudaStatus = cudaMalloc(&C_dev_arr, bench_len * sizeof(Ctype));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc (C_dev_arr) failed!\n");
        is_successful = false;
        goto Error;
    }

    cudaStatus = cudaMemcpy(A_dev_arr, A_host_arr, bench_len * sizeof(Atype), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (A_arrs) failed!\n");
        is_successful = false;
        goto Error;
    }

    cudaStatus = cudaMemcpy(B_dev_arr, B_host_arr, bench_len * sizeof(Btype), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy (B_arrs) failed!\n");
        is_successful = false;
        goto Error;
    }

    //run_kernels!
    //---------------------------------------------------------------------------------------------------------------------------------
    for(size_t i = 0; i < num_of_kernels; i++)
    {
        auto f = func_vec[i].second;
        auto message = func_vec[i].first;

        std::cout << "Launching kernel: "  << message << std::endl;

        f(A_dev_arr, B_dev_arr, C_dev_arr, bench_len);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            is_successful = false;
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            is_successful = false;
            goto Error;
        }

        cudaStatus = cudaMemcpy(results[i], C_dev_arr, bench_len * sizeof(Ctype), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaMemcpy (C_arrs) failed!");
            is_successful = false;
            goto Error;
        }
    }

Error:
    cudaFree(A_dev_arr);
    cudaFree(B_dev_arr);
    cudaFree(C_dev_arr);

    return is_successful;
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

//We require a bunch of comparison functions (they are all implemented in host_funcs.cpp)

bool equal_host(const uint256_g& a, const uint256_g& b);
bool equal_host(const uint512_g& a, const uint512_g& b);
bool equal_proj_host(const ec_point& a, const ec_point& b);
bool equal_jac_host(const ec_point& a, const ec_point& b);

//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------

//check addition test

using general_func_vec_t = kernel_func_vec_t<uint256_g, uint256_g, uint256_g>;

void add_uint256_naive_driver(uint256_g*, uint256_g*, uint256_g*, size_t);
void add_uint256_asm_driver(uint256_g*, uint256_g*, uint256_g*, size_t);

general_func_vec_t addition_bench = {
    {"naive approach", add_uint256_naive_driver},
	{"asm", add_uint256_asm_driver}
};

static constexpr size_t  FIXED_SIZE_ADD_TEST_SIZE = 6;
static constexpr std::vector<std::string> FIXED_SIZE_ADD_DATA = {};

TEST_CASE( "fixed_size addtion" , "[basic]" )
{
    static constexpr std::vector<std::string> data = FIXED_SIZE_ADD_DATA;
    static constexpr size_t test_size = FIXED_SIZE_ADD_TEST_SIZE;
    
    using A_type = uint256_g;
    using B_type = uintt256_g;
    using C_type = uint256_g;
    auto& func_vec = addition_bench;

    auto num_of_kernels = func_vec.size(); 

    A_type[test_size] a_arr;
    B_type[test_size] b_arr;
    C_type[test_size] c_arr;

    std::vector<C_type*> results_ptr;
    std::vector<C_type> results;
    results.reserve(num_of_kernels * test_size);

    for (size_t i = 0; i < num_of_kernels; i++)
    {
        result_ptr.push_back(results.data() + i * test_size);
    }

    bool flag = test_framework(unc_vec, a_arr, b_arr, c_arr, results, data, test_len);
    REQUIRE(flag);

    for (size_t i = 0; i < num_of_kernels; i++)
    {
        auto message = func_vec[i].first;
        INFO( "Checking kernel: " << message);
        
        bool test_passed = true;
        for (size_t j=0; j < test_len; j++)
        {
            CHECK(equal_host(results[i][j], c_arr[j]));
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------------------





