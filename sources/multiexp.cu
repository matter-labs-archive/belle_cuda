#include "cuda_structs.h"

#include <iostream>

//Various algorithms for simultaneous multiexponentiation: naive approach and Pippenger algorithm
//naive approach was widely inspired by https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------

//There are four versions using naive approach:
//1) using warp level reduction and atomics
//2) using block level reduction and atomics
//3) using block level reduction and recursion

//TODO: it seems that the best way is to combine these approaches, e.g. do several levels of atomic add, than block reduce - there is a vast field
//for experiements

//TODO: implement using warp level reduction and recursion

//TODO: implement version with cooperative groups

//TODO: implement approach using CUB library: http://nvlabs.github.io/cub/index.html

//we have implemented vectorized loads inspired by: https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

//Useful miscellaneous functions
//-----------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC inline void __shfl_down(const ec_point& in_var, ec_point& out_var, unsigned int offset, int width=32)
{
    //ec_point = 3 * 8  = 24 int = 6 int4
    const int4* a = reinterpret_cast<const int4*>(&in_var);
    int4* b = reinterpret_cast<int4*>(&out_var);

    for (unsigned i = 0; i < 6; i++)
    {
        b[i].x = __shfl_down_sync(0xFFFFFFFF, a[i].x, offset, width);
        b[i].y = __shfl_down_sync(0xFFFFFFFF, a[i].y, offset, width);
        b[i].z = __shfl_down_sync(0xFFFFFFFF, a[i].z, offset, width);
        b[i].w = __shfl_down_sync(0xFFFFFFFF, a[i].w, offset, width);
    }
}

DEVICE_FUNC inline ec_point warpReduceSum(ec_point val)
{
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
    { 
        ec_point temp;
        __shfl_down(val, temp, offset);
        val = ECC_ADD(val, temp);
    }
           
    return val;
}

DEVICE_FUNC inline ec_point blockReduceSum(ec_point val)
{
    // Shared mem for 32 partial sums
    static __shared__ ec_point shared[WARP_SIZE]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Each warp performs partial reduction
    val = warpReduceSum(val);     

    // Write reduced value to shared memory
    if (lane==0)
        shared[wid]=val; 

    // Wait for all partial reductions
    __syncthreads();              

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : point_at_infty();

    //Final reduce within first warp
    if (wid == 0)
        val = warpReduceSum(val); 

    return val;
}


//1) using warp level reduction and atomics
//---------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_warp_level_atomics(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len, int* mutex)
{
	ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x;
	}

    acc = warpReduceSum(acc);
 
    if ((threadIdx.x & (warpSize - 1)) == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);
          
        *out = ECC_ADD(*out, acc);
       
        atomicExch(mutex, 0);
    }  
}

void naive_multiexp_kernel_warp_level_atomics_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
  	int minGridSize;
  	int realGridSize;

    int* mutex;
    cudaMalloc((void**)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_warp_level_atomics, 0, 0);
    realGridSize = (arr_len + blockSize - 1) / blockSize;

    int maxActiveBlocks;
    cudaDeviceProp prop;
  	cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;
	cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, naive_multiexp_kernel_warp_level_atomics, blockSize, 0);
    if (error == cudaSuccess && realGridSize > maxActiveBlocks * smCount)
    	realGridSize = maxActiveBlocks * smCount;

	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;

    //create point at infty and copy it to output arr

    ec_point point_at_infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };

    cudaMemcpy(out, &point_at_infty, sizeof(ec_point), cudaMemcpyHostToDevice);

	naive_multiexp_kernel_warp_level_atomics<<<realGridSize, blockSize>>>(point_arr, power_arr, out, arr_len, mutex);

    cudaFree(mutex);
}

//2) using block level reduction and atomics
//---------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_block_level_atomics(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len, int* mutex)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x;
	}

    acc = blockReduceSum(acc);
    if (threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0);
        *out = ECC_ADD(*out, acc);
        atomicExch(mutex, 0);  
    }
}

void naive_multiexp_kernel_block_level_atomics_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len)
{
	int blockSize;
    int minGridSize;
  	int realGridSize;

    int* mutex;
    cudaMalloc((void**)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_atomics, 4 * N * 3 * WARP_SIZE, 0);
  	realGridSize = (arr_len + blockSize - 1) / blockSize;

    int maxActiveBlocks;
    cudaDeviceProp prop;
  	cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;
	cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, naive_multiexp_kernel_block_level_atomics, 
        blockSize, 4 * N * 3 * WARP_SIZE);
    if (error == cudaSuccess && realGridSize > maxActiveBlocks * smCount)
    	realGridSize = maxActiveBlocks * smCount;

    ec_point point_at_infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };

    cudaMemcpy(out, &point_at_infty, sizeof(ec_point), cudaMemcpyHostToDevice);

	std::cout << "Grid size: " << realGridSize << ",  min grid size: " << minGridSize << ",  blockSize: " << blockSize << std::endl;
	naive_multiexp_kernel_block_level_atomics<<<realGridSize, blockSize>>>(point_arr, power_arr, out, arr_len, mutex);

    cudaFree(mutex);
}

//3) using block level reduction and recursion
//---------------------------------------------------------------------------------------------------------------------------------------------------------------

__global__ void naive_multiexp_kernel_block_level_recursion(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        ec_point x = ECC_EXP(point_arr[tid], power_arr[tid]);
        acc = ECC_ADD(acc, x);
        tid += blockDim.x * gridDim.x;
	}

    acc = blockReduceSum(acc);
    
    if (threadIdx.x == 0)
        out_arr[blockIdx.x] = acc;
}

__global__ void naive_kernel_block_level_reduction(ec_point* in_arr, ec_point* out_arr, size_t arr_len)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
    {   
        acc = ECC_ADD(acc, in_arr[tid]);
        tid += blockDim.x * gridDim.x;
	}

    acc = blockReduceSum(acc);

    if (threadIdx.x == 0)
        out_arr[blockIdx.x] = acc;
}

void naive_multiexp_kernel_block_level_recursion_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;
    int maxActiveBlocks;

    int maxExpGridSize, ExpBlockSize;
    int maxReductionGridSize, ReductionBlockSize;
    
    const size_t SHARED_MEMORY_USED = 4 * N * 3 * WARP_SIZE;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

    //first we find the optimal geometry for both kernels: exponentialtion kernel and reduction

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_recursion, SHARED_MEMORY_USED, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, naive_multiexp_kernel_block_level_recursion, blockSize, SHARED_MEMORY_USED);
    maxExpGridSize = maxActiveBlocks * smCount;
    ExpBlockSize = blockSize;

    //the same routine for reduction phase

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_kernel_block_level_reduction, SHARED_MEMORY_USED, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, naive_kernel_block_level_reduction, blockSize, SHARED_MEMORY_USED);
    maxReductionGridSize = maxActiveBlocks * smCount;
    ReductionBlockSize = blockSize;

    //do the first stage:

    realGridSize = (arr_len + ExpBlockSize - 1) / ExpBlockSize;
    realGridSize = min(realGridSize, maxExpGridSize);

	std::cout << "Real grid size: " << realGridSize << ",  blockSize: " << ExpBlockSize << std::endl;
	naive_multiexp_kernel_block_level_recursion<<<realGridSize, ExpBlockSize>>>(point_arr, power_arr, out_arr, arr_len);

    //NB: we also need to use temporary array (we need to store all temporary values somewhere!)

    ec_point* d_temp_storage = nullptr;

    if (realGridSize > 1)
    {
        cudaMalloc((void **)&d_temp_storage, realGridSize * sizeof(ec_point));
    }

    arr_len = realGridSize;
    ec_point* temp_input_arr = out_arr;
    ec_point* temp_output_arr = d_temp_storage;
    unsigned iter_count = 0;
 
    while (arr_len > 1)
    {
        cudaDeviceSynchronize();
        realGridSize = (arr_len + ReductionBlockSize - 1) / ReductionBlockSize;
        realGridSize = min(realGridSize, maxReductionGridSize);

        std::cout << "iter " << ++iter_count << ", real grid size: " << realGridSize << ",  blockSize: " << ReductionBlockSize << std::endl;
        naive_kernel_block_level_reduction<<<realGridSize, ReductionBlockSize>>>(temp_input_arr, temp_output_arr, arr_len);
        arr_len = realGridSize;

        //swap arrays
        ec_point* swapper = temp_input_arr;
        temp_input_arr = temp_output_arr;
        temp_output_arr = swapper;
    }

    //copy output to the correct array! (but we are just moving pointers :)
    if (out_arr != temp_input_arr)
        cudaMemcpy(out_arr, temp_input_arr, sizeof(ec_point), cudaMemcpyDeviceToDevice);
    cudaFree(d_temp_storage);
}

//Pippenger
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------

//the main stage of Pippenger algorithm is splitting a lot op points among a relatively small amount of chunks (or bins)
//This operation can be considered as a sort of histogram construction, so we can use specific Cuda algorithms.
//Source of inspiration are: 
//https://devblogs.nvidia.com/voting-and-shuffling-optimize-atomic-operations/
//https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
//https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

//TODO: what is the exact difference between inline and __inline__

#define SMALL_C 8
#define SMALL_CHUNK_SIZE 256
#define MAX_POWER_BITLEN 256

DEVICE_FUNC __inline__ void __shfl(const ec_point& in_var, ec_point& out_var, unsigned int mask, unsigned int offset, int width=32)
{
    //ec_point = 3 * 8  = 24 int = 6 int4
    const int4* a = reinterpret_cast<const int4*>(&in_var);
    int4* b = reinterpret_cast<int4*>(&out_var);

    for (unsigned i = 0; i < 6; i++)
    {
        b[i].x = __shfl_sync(mask, a[i].x, offset, width);
        b[i].y = __shfl_sync(mask, a[i].y, offset, width);
        b[i].z = __shfl_sync(mask, a[i].z, offset, width);
        b[i].w = __shfl_sync(mask, a[i].w, offset, width);
    }
}

DEVICE_FUNC __inline__ uint32_t get_peers(uint32_t key)
{
    uint32_t peers=0;
    bool is_peer;

    // in the beginning, all lanes are available
    uint32_t unclaimed=0xffffffff;

    do
    {
        // fetch key of first unclaimed lane and compare with this key
        is_peer = (key == __shfl_sync(unclaimed, key, __ffs(unclaimed) - 1));

        // determine which lanes had a match
        peers = __ballot_sync(unclaimed, is_peer);

        // remove lanes with matching keys from the pool
        unclaimed ^= peers;


    }
    // quit if we had a match
    while (!is_peer);

    return peers;
}

//returns the index of leader peer

DEVICE_FUNC __inline__ uint reduce_peers(uint peers, ec_point& pt)
{
    int lane = threadIdx.x & (warpSize - 1);

    // find the peer with lowest lane index
    uint first = __ffs(peers)-1;

    // calculate own relative position among peers
    int rel_pos = __popc(peers << (32 - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while(__any_sync(0xffffffff, peers))
    {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl() only works if both threads participate, so we always do.
        ec_point temp;
        __shfl(pt, temp, 0xffffffff, next - 1);

        // only add if there was anything to add
        if (next)
        {
            pt = ECC_ADD(pt, temp);
        }

        // all lanes with their least significant index bit set are done
        uint32_t done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~ __ballot_sync(0xffffffff, done);

        // abuse relative position as iteration counter
        rel_pos >>= 1;
    }

    return first;
}


struct Lock
{
    int mutex;

    Lock() {} 
    ~Lock() = default;

    DEVICE_FUNC void init()
    {
        mutex = 0;
    }
    
    DEVICE_FUNC void lock()
    {
        while (atomicCAS(&mutex, 0, 1) != 0);
    }

    DEVICE_FUNC bool try_lock()
    {
        return (atomicExch(&mutex, 1) == 0);
    }

    DEVICE_FUNC void unlock()
    {
        atomicExch(&mutex, 0);
    }
};

struct Bin
{
    Lock lock;
    ec_point pt;
};

DEVICE_FUNC __inline__ uint get_key(const uint256_g& val, uint chunk_num, uint bitwidth)
{
    uint bit_pos = chunk_num * bitwidth;
    uint limb_idx = bit_pos / BITS_PER_LIMB;
    uint offset = bit_pos % BITS_PER_LIMB;

    uint low_part = val.n[limb_idx];
    uint high_part = (limb_idx < N - 1 ? val.n[limb_idx + 1] : 0);

    uint res = __funnelshift_r(low_part, high_part, offset);
    res &= (1 << bitwidth) - 1;

    return res;   
}

DEVICE_FUNC __inline__ ec_point from_affine_point(const affine_point& pt)
{
    return ec_point{pt.x, pt.y, BASE_FIELD_R};
}

#define REVERSE_INDEX(index, max_index) (max_index - index - 1)

DEVICE_FUNC __inline__ void block_level_histo(affine_point* pt_arr, uint256_g* power_arr, 
    size_t arr_start_pos, size_t arr_end_pos, uint chunk_num, ec_point* out_histo_arr)
{
    //we exclude the bin corresponding to value 0
    
    __shared__ Bin bins[SMALL_CHUNK_SIZE]; 
    uint lane = threadIdx.x % WARP_SIZE;
    
    //first we need to init all bins

    size_t idx = threadIdx.x;
    while (idx < SMALL_CHUNK_SIZE)
    {
        bins[idx].lock.init();
        bins[idx].pt = point_at_infty();
        idx += blockDim.x;
    }

    __syncthreads();

    idx = arr_start_pos + threadIdx.x;
    while (idx < arr_end_pos)
    {
        ec_point pt = from_affine_point(pt_arr[idx]);
        uint key = get_key(power_arr[idx], chunk_num, SMALL_C);

        uint peers = get_peers(key);
        uint leader = reduce_peers(peers, pt);
        
        if (lane == leader)
        {
            uint real_key = REVERSE_INDEX(key, SMALL_CHUNK_SIZE);

            bool leaveLoop = false;
            while (!leaveLoop)
            {
                if (bins[real_key].lock.try_lock())
                {
                    //critical section
                    bins[real_key].pt = ECC_ADD(pt, bins[real_key].pt);
                    leaveLoop = true;
                    bins[real_key].lock.unlock();
                    
                }
                __threadfence_block();
            }
            
        }

        idx += blockDim.x;
    }

    __syncthreads();

    idx = threadIdx.x;
    while (idx < SMALL_CHUNK_SIZE)
    {
        out_histo_arr[idx] = bins[idx].pt;
        idx += blockDim.x;
    }
}

__global__ void device_level_histo(affine_point* pt_arr, uint256_g* power_arr, ec_point* out_histo, size_t arr_len, uint BLOCKS_PER_BIN)
{  
    uint chunk_num = blockIdx.x / BLOCKS_PER_BIN;

    uint ELEMS_PER_BLOCK = (arr_len + BLOCKS_PER_BIN - 1) / BLOCKS_PER_BIN;
    size_t start_pos = (blockIdx.x % BLOCKS_PER_BIN) * ELEMS_PER_BLOCK;
    size_t end_pos = min(start_pos + ELEMS_PER_BLOCK, arr_len);
    
    size_t output_pos = SMALL_CHUNK_SIZE *  blockIdx.x;

    block_level_histo(pt_arr, power_arr, start_pos, end_pos, chunk_num, out_histo + output_pos);
}

DEVICE_FUNC __inline__ void block_level_histo_shrinking(const ec_point* histo_arr, ec_point* shrinked_histo, 
    size_t arr_start_pos, size_t arr_end_pos)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x + arr_start_pos;
	while (tid < arr_end_pos)
	{   
        acc = ECC_ADD(acc, histo_arr[tid]);
        tid += blockDim.x;
	}

    shrinked_histo[threadIdx.x] = acc;
}

__global__ void shrink_histo(const ec_point* local_histo_arr, ec_point* shrinked_histo, size_t BLOCKS_PER_BIN)
{
    uint ELEMS_PER_BLOCK = SMALL_CHUNK_SIZE * BLOCKS_PER_BIN;
    size_t start_pos = blockIdx.x * ELEMS_PER_BLOCK;
    size_t end_pos = start_pos + ELEMS_PER_BLOCK;
    
    size_t output_pos = SMALL_CHUNK_SIZE *  blockIdx.x;

    block_level_histo_shrinking(local_histo_arr, shrinked_histo + output_pos, start_pos, end_pos);
} 

//Another important part of Pippenger algorithm is the evaluation of Rolling sum - we use the combination of scan + reduction primitives
//We do also try to avoid bank conflicts (although it sounds like a sort of some weird magic)

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ( ((n) >> NUM_BANKS) + ( (n) >> (2 * LOG_NUM_BANKS)) )
#else
#define CONFLICT_FREE_OFFSET(n)  ((n) >> LOG_NUM_BANKS) 
#endif


//NB: arr_len should be a power of two
//TBD: implement conflict free offsets (correctly!)

__global__ void scan_and_reduce(const ec_point* global_in_arr, ec_point* out)
{
    // allocated on invocation
    __shared__ ec_point temp[SMALL_CHUNK_SIZE * 2];

    //scanning

    uint tid = threadIdx.x;
    uint offset = 1;
    const ec_point* in_arr = global_in_arr + blockIdx.x * SMALL_CHUNK_SIZE;

    uint ai = tid;
    uint bi = tid + (SMALL_CHUNK_SIZE / 2);

    uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint bankOffsetB = CONFLICT_FREE_OFFSET(ai);
    temp[ai + bankOffsetA] = in_arr[ai];
    temp[bi + bankOffsetB] = in_arr[bi]; 

    // build sum in place up the tree
    for (int d = SMALL_CHUNK_SIZE >> 1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset*(2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 

            temp[bi] = ECC_ADD(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    
    if (tid == 0)
    {
        temp[SMALL_CHUNK_SIZE - 1 + CONFLICT_FREE_OFFSET(SMALL_CHUNK_SIZE - 1)] = point_at_infty();
    }

    // traverse down tree & build scan
    for (uint d = 1; d < SMALL_CHUNK_SIZE; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 
            
            ec_point t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = ECC_ADD(t, temp[bi]);
        }
    }
    
    __syncthreads();
   
    //reducing

    for (int d = SMALL_CHUNK_SIZE >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
            temp[tid] = ECC_ADD(temp[tid], temp[tid + d]);
        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x * SMALL_CHUNK_SIZE] = temp[0];
}

//the last kernel is not important, however it is very useful for debugging purposes

__global__ void final_reduce(ec_point* arr)
{
    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / SMALL_C;
    
    __shared__ ec_point temp[NUM_OF_CHUNKS];
    
    uint tid = threadIdx.x;
    ec_point val = (tid < NUM_OF_CHUNKS ? arr[tid * SMALL_CHUNK_SIZE] : point_at_infty());

    for (int j = 0; j < tid * SMALL_C; j++)
        val = ECC_DOUBLE(val);

    temp[tid] = val;

    __syncthreads();

    for (int d = NUM_OF_CHUNKS >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
            temp[tid] = ECC_ADD(temp[tid], temp[tid + d]);
        __syncthreads();
    }
    
    if (tid == 0)
        arr[0] = temp[0];
}

//Pippenger: basic version - simple, yet powerful. The same version of Pippenger algorithm is implemented in libff and Bellman

void small_Pippenger_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
  	//NB: gridSize should be a power of two
    uint gridSize;
    int maxActiveBlocks;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

    //we start by finding the optimal geometry and grid size

    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / SMALL_C;
    uint SHARED_MEMORY_USED = 4 * 8 * 3 * SMALL_CHUNK_SIZE;

    gridSize = (arr_len + SMALL_CHUNK_SIZE - 1) / SMALL_CHUNK_SIZE;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, device_level_histo, SMALL_CHUNK_SIZE, SHARED_MEMORY_USED);
    gridSize = min(maxActiveBlocks * smCount, gridSize);
    gridSize = max(gridSize, NUM_OF_CHUNKS);

    //find the closest power of 2

    uint num = BITS_PER_LIMB - __builtin_clz(gridSize) - 1;
    gridSize = (1 << num);

    //-----------------------------------------------------------------------------------------------------------------
    //calculate kernel run parameteres

    gridSize = 64 * 2;
 
    uint BLOCKS_PER_BIN = gridSize / NUM_OF_CHUNKS;
    uint ELEMS_PER_BLOCK = (arr_len + BLOCKS_PER_BIN - 1) / BLOCKS_PER_BIN;

    std::cout << "Num of chunks : " << NUM_OF_CHUNKS << ", blocks per bin : " << BLOCKS_PER_BIN << 
        ", elems per block : " << ELEMS_PER_BLOCK << std::endl;

    //allocate memory for temporary array of needed

    ec_point* histo_arr = nullptr;
    size_t HISTO_ELEMS_COUNT = SMALL_CHUNK_SIZE * gridSize;
    cudaMalloc((void **)&histo_arr, HISTO_ELEMS_COUNT * sizeof(ec_point));
    
    //run kernels - one after after another:
    //1) collect local block-level histograms
    //2) shrink all local histograms to a larger one
    //3) perform block level scan and reduce on shrinked histogram

    device_level_histo<<<gridSize, SMALL_CHUNK_SIZE>>>(point_arr, power_arr, histo_arr, arr_len, BLOCKS_PER_BIN);
   
    cudaDeviceSynchronize();

    shrink_histo<<<NUM_OF_CHUNKS, SMALL_CHUNK_SIZE>>>(histo_arr, out_arr, BLOCKS_PER_BIN);
    cudaDeviceSynchronize();

    //TODO: try PIPPENGER_BLOCK_SIZE / 2
    scan_and_reduce<<<NUM_OF_CHUNKS, SMALL_CHUNK_SIZE / 2>>>(out_arr, out_arr);

    //for debugging and tests only!
    cudaDeviceSynchronize();
    final_reduce<<<1, NUM_OF_CHUNKS>>>(out_arr);

    cudaFree(histo_arr);
}


//versoin of Pippenger algorithm with large bins
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------

#define LARGE_C 16
#define LARGE_CHUNK_SIZE 65536
#define SCAN_BLOCK_SIZE 256

struct kernel_geometry
{
    int gridSize;
    int blockSize;
};

template<typename T>
kernel_geometry find_optimal_geometry(T func, uint shared_memory_used, uint32_t smCount)
{
    int gridSize;
    int blockSize;
    int maxActiveBlocks;

    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, func, shared_memory_used, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, func, blockSize, shared_memory_used);
    gridSize = maxActiveBlocks * smCount;

    return kernel_geometry{gridSize, blockSize};
}

//first step: histogramming
//----------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC __inline__ void large_histo_impl(affine_point* pt_arr, uint256_g* power_arr, 
    size_t arr_start_pos, size_t arr_end_pos, uint chunk_num, ec_point* bins, Lock* locks)
{
    uint idx = arr_start_pos + threadIdx.x;
    while (idx < arr_end_pos)
    {
        affine_point& pt = pt_arr[idx];
        uint key = get_key(power_arr[idx], chunk_num, LARGE_C);

        uint real_key = REVERSE_INDEX(key, LARGE_CHUNK_SIZE);

        bool leaveLoop = false;
        while (!leaveLoop)
        {
            if ( locks[real_key].try_lock())
            {
                //critical section
                bins[real_key] = ECC_MIXED_ADD(bins[real_key], pt);
                leaveLoop = true;
                locks[real_key].unlock();
                
            }
            __threadfence();
        }

        idx += blockDim.x;
    }
}

__global__ void init_large_histo(ec_point* bins, Lock* locks)
{
    uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < LARGE_CHUNK_SIZE * NUM_OF_CHUNKS)
    {
        locks[idx].init();
        bins[idx] = point_at_infty();
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void large_histo(affine_point* pt_arr, uint256_g* power_arr, ec_point* bins, Lock* locks, size_t arr_len, uint BLOCKS_PER_BIN)
{   
    uint chunk_num = blockIdx.x / BLOCKS_PER_BIN;
    uint ELEMS_PER_BLOCK = (arr_len + BLOCKS_PER_BIN - 1) / BLOCKS_PER_BIN;
    size_t start_pos = (blockIdx.x % BLOCKS_PER_BIN) * ELEMS_PER_BLOCK;
    size_t end_pos = min(start_pos + ELEMS_PER_BLOCK, arr_len);
    size_t output_pos = LARGE_CHUNK_SIZE * chunk_num;
    
    large_histo_impl(pt_arr, power_arr, start_pos, end_pos, chunk_num, bins + output_pos, locks + output_pos);
}

void BUILD_HISTOGRAM(affine_point* pt_arr, uint256_g* power_arr, ec_point* bins, Lock* locks, size_t arr_len, uint32_t smCount)
{
    {
        kernel_geometry geometry = find_optimal_geometry(init_large_histo, 0, smCount);
        init_large_histo<<<geometry.gridSize, geometry.blockSize>>>(bins, locks);
        cudaDeviceSynchronize();
    }
       
    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    
    kernel_geometry geometry = find_optimal_geometry(large_histo, 0, smCount);
    int minGridSize = geometry.gridSize;
    int blockSize = geometry.blockSize;

    int gridSize = max(minGridSize, NUM_OF_CHUNKS);

    //find the closest power of 2

    uint num = BITS_PER_LIMB - __builtin_clz(gridSize) - 1;
    gridSize = (1 << num);
 
    uint BLOCKS_PER_BIN = gridSize / NUM_OF_CHUNKS;

    large_histo<<<gridSize, blockSize>>>(pt_arr, power_arr, bins, locks, arr_len, BLOCKS_PER_BIN);
    cudaDeviceSynchronize();
}

//second step: scanning
//------------------------------------------------------------------------------------------------------------------------------------------------

//NB: temp array size is twice as large as blocksize;

DEVICE_FUNC void block_level_prescan(const ec_point* in_arr, ec_point* out_arr, ec_point* block_sums)
{
    //TODO: temp array size is too small
    __shared__ ec_point temp[SCAN_BLOCK_SIZE + SCAN_BLOCK_SIZE / 2];
    
    uint tid = threadIdx.x;
    uint offset = 1;
   
    uint ai = tid;
    uint bi = tid + (SCAN_BLOCK_SIZE / 2);

    uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint bankOffsetB = CONFLICT_FREE_OFFSET(ai);
    temp[ai + bankOffsetA] = in_arr[ai];
    temp[bi + bankOffsetB] = in_arr[bi]; 

    // build sum in place up the tree
    for (int d = SCAN_BLOCK_SIZE >> 1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            uint bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 

            temp[bi] = ECC_ADD(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
  
    if (tid == 0)
    {
        temp[SCAN_BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(SCAN_BLOCK_SIZE - 1)] = point_at_infty();
    }

    // traverse down tree & build scan
    for (uint d = 1; d < SCAN_BLOCK_SIZE; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            uint ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 
            
            ec_point t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = ECC_ADD(t, temp[bi]);
        }
    }
    
    __syncthreads();

     //print total sum to array of block sums
   
    if (tid == 0 && block_sums)
        block_sums[blockIdx.x] = ECC_ADD(temp[SCAN_BLOCK_SIZE - 1 + CONFLICT_FREE_OFFSET(SCAN_BLOCK_SIZE - 1)], in_arr[SCAN_BLOCK_SIZE - 1]);

    __syncthreads();
   
    //print result back to global memory

    if (tid < SCAN_BLOCK_SIZE)
    {
        ai = tid;
        bi = tid + (SCAN_BLOCK_SIZE/ 2);
        bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        bankOffsetB = CONFLICT_FREE_OFFSET(ai);

        out_arr[ai] = temp[ai + bankOffsetA];
        out_arr[bi] = temp[bi + bankOffsetB];
    }
}

__global__ void prescan(const ec_point* global_in_arr, ec_point* global_out_arr, ec_point* block_sums)
{
    const ec_point* in_arr = global_in_arr + blockIdx.x * SCAN_BLOCK_SIZE;
    ec_point* out_arr = global_out_arr + blockIdx.x * SCAN_BLOCK_SIZE;

    block_level_prescan(in_arr, out_arr, block_sums);
}

//TODO: what is the best possible algortihm got adjusting increments?

__global__ void adjust_incr(ec_point* in_arr, ec_point* out_arr, ec_point* incr_arr, size_t num_of_elems)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < num_of_elems)
    {   
        out_arr[tid] = ECC_ADD(in_arr[tid], incr_arr[tid / SCAN_BLOCK_SIZE]);
        tid += blockDim.x * gridDim.x;     
	}
}

void SCAN_ARRAY(ec_point* arr, ec_point* block_sums, uint smCount)
{
    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    constexpr uint num_of_elems = LARGE_CHUNK_SIZE * NUM_OF_CHUNKS;
    constexpr uint NUM_OF_BLOCKS = NUM_OF_CHUNKS * (LARGE_CHUNK_SIZE / SCAN_BLOCK_SIZE);

    prescan<<<NUM_OF_BLOCKS, SCAN_BLOCK_SIZE / 2>>>(arr, arr, block_sums);
    cudaDeviceSynchronize();
    prescan<<<NUM_OF_BLOCKS / SCAN_BLOCK_SIZE, SCAN_BLOCK_SIZE / 2>>>(block_sums, block_sums, nullptr);
    cudaDeviceSynchronize();

    kernel_geometry geometry = find_optimal_geometry(adjust_incr, 0, smCount);
    adjust_incr<<<geometry.gridSize, geometry.blockSize>>>(arr, arr, block_sums, num_of_elems);
    cudaDeviceSynchronize();
}

//third step: reducing
//------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC void reduce_chunk(const ec_point* point_arr, ec_point* out, Lock* lock, size_t arr_len)
{
    ec_point acc = point_at_infty();
    //printf("%d\n", arr_len);
    
    size_t tid = threadIdx.x;
	while (tid < arr_len)
	{   
        acc = ECC_ADD(acc, point_arr[tid]);
        tid += blockDim.x;
	}

    acc = blockReduceSum(acc);
    if (threadIdx.x == 0)
    {
        lock->lock();
        *out = ECC_ADD(*out, acc);
        lock->unlock();
    }
}

__global__ void init_reduce(ec_point* arr, size_t arr_len)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < arr_len)
    {
        arr[idx] = point_at_infty();
        idx += blockDim.x * gridDim.x;
    }
}

//NB: blocks per bin should be a power of 2


__global__ void reduce(ec_point* in_arr, ec_point* out_arr, Lock* locks, uint BLOCKS_PER_BIN)
{
    uint chunk_num = blockIdx.x / BLOCKS_PER_BIN;
    uint ELEMS_PER_BLOCK = LARGE_CHUNK_SIZE / BLOCKS_PER_BIN;
    size_t start_pos = blockIdx.x * ELEMS_PER_BLOCK;

    reduce_chunk(in_arr + start_pos, out_arr + chunk_num, locks + chunk_num, ELEMS_PER_BLOCK);
}

void REDUCE_ARRAY(ec_point* in_arr, ec_point* out_arr, Lock* locks, uint smCount)
{
    {
        kernel_geometry geometry = find_optimal_geometry(init_reduce, 0, smCount);
        uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
        init_reduce<<<geometry.gridSize, geometry.blockSize>>>(out_arr, LARGE_CHUNK_SIZE * NUM_OF_CHUNKS);
        cudaDeviceSynchronize();
    }
    
    //TODO: calculate shared_memory usage
    const size_t SHARED_MEMORY_USED = 4 * N * 3 * WARP_SIZE;

    kernel_geometry geometry = find_optimal_geometry(reduce, SHARED_MEMORY_USED, smCount);
    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    
    int minGridSize = geometry.gridSize;
    int blockSize = geometry.blockSize;

    uint gridSize = max(minGridSize, NUM_OF_CHUNKS);
    //find the closest power of 2

    uint num = BITS_PER_LIMB - __builtin_clz(gridSize) - 1;
    gridSize = (1 << num);
 
    uint BLOCKS_PER_BIN = gridSize / NUM_OF_CHUNKS;

    reduce<<<gridSize, blockSize>>>(in_arr, out_arr, locks, BLOCKS_PER_BIN);
    cudaDeviceSynchronize();
}

//fourth step: shrinking (not important, this step is better done on CPU, we implement it for debugging purposes only)
//----------------------------------------------------------------------------------------------------------------------------------------------

__global__ void final_shrinking(ec_point* arr, ec_point* total, size_t gap)
{
    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    
    __shared__ ec_point temp[NUM_OF_CHUNKS];
    
    uint tid = threadIdx.x;
    ec_point val = (tid < NUM_OF_CHUNKS ? arr[tid * gap] : point_at_infty());

    for (int j = 0; j < tid * LARGE_C; j++)
        val = ECC_DOUBLE(val);

    temp[tid] = val;

    __syncthreads();

    for (int d = NUM_OF_CHUNKS >> 1; d > 0; d >>= 1)
    {
        if (tid < d)
            temp[tid] = ECC_ADD(temp[tid], temp[tid + d]);
        __syncthreads();
    }

    if (tid == 0)
    {
        //dirty Hack: delete it - we transform this point tp affine
        ec_point res = temp[0];
        uint256_g z_inv = FIELD_MUL_INV(res.z);
        res.x = MONT_MUL(res.x, z_inv);
        res.y = MONT_MUL(res.y, z_inv);
        res.z = BASE_FIELD_R;

        *total = res;
    }
}

//Large Pippenger driver
//---------------------------------------------------------------------------------------------------------------------------------------------------

//TODO: Check scanning for the whole range)

void large_Pippenger_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len)
{
    //TODO: do we really need CUDA device synchronize?
    
    Lock* locks = nullptr;
    ec_point* temporary_array = nullptr;
    ec_point* out_temp_arr = nullptr;


    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    
    cudaError_t cudaStatus = cudaMalloc((void **)&locks, NUM_OF_CHUNKS * LARGE_CHUNK_SIZE * sizeof(Lock));
    if (cudaStatus != cudaSuccess)
        std::cout << "Error!" << std::endl;

    cudaStatus = cudaMalloc((void **)&temporary_array, NUM_OF_CHUNKS * LARGE_CHUNK_SIZE * sizeof(ec_point));
    if (cudaStatus != cudaSuccess)
        std::cout << "Error!" << std::endl;
    cudaMalloc((void **)&out_temp_arr, NUM_OF_CHUNKS * LARGE_CHUNK_SIZE * sizeof(ec_point));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

    BUILD_HISTOGRAM(point_arr, power_arr, temporary_array, locks, arr_len, smCount);
    SCAN_ARRAY(temporary_array, out_temp_arr, smCount);
    //TODO: there is no need for additional array allocation!
    
    REDUCE_ARRAY(temporary_array, out_temp_arr, locks, smCount);
    //cudaMemcpy(out, out_temp_arr, 65536 * sizeof(ec_point), cudaMemcpyDeviceToDevice);
    
    final_shrinking<<<1, NUM_OF_CHUNKS>>>(out_temp_arr, out, 1);
    
    cudaFree(locks);
    cudaFree(temporary_array);
    cudaFree(out_temp_arr);
}


//Version of Pippenger algorithm based on sorting networks and additional memory
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------------------------------------------

//first step: sorting
//---------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC __inline__ void exchange_points(affine_point& x, affine_point& y)
{
    //affine_point = 2 * 8  = 16 int = 4 int4
    int4* a = reinterpret_cast<int4*>(&x);
    int4* b = reinterpret_cast<int4*>(&y);

    for (unsigned i = 0; i < 4; i++)
    {
        int4 swapper;
        swapper = a[i];
        a[i] = b[i];
        b[i] = swapper;
    }
}

DEVICE_FUNC __inline__ void copy_point(const affine_point* source, affine_point* dest)
{
    //affine_point = 2 * 8  = 16 int = 4 int4
    const int4* a = reinterpret_cast<const int4*>(source);
    int4* b = reinterpret_cast<int4*>(dest);

    for (unsigned i = 0; i < 4; i++)
    {
        b[i] = a[i];
    }
}

DEVICE_FUNC __inline__ void compare_exchange_points(affine_point& x, affine_point& y, uint key_x, uint key_y, bool is_ascending_order)
{
    bool exchange = (key_x > key_y) ^ is_ascending_order;
    if (exchange)
        exchange_points(x, y); 
}

//NB: We are going to pad array length to 64 elem boundary: it should be done in main 

DEVICE_FUNC __inline__ void bitonic_sort_step(affine_point* values, const uint256_g* powers, size_t arr_len, size_t j, size_t k, uint chunk_num)
{
    /* Sorting partners: i and ixj */
    unsigned int i, ixj; 
    size_t idx = (threadIdx.x + blockIdx.x * blockDim.x) % arr_len;
    i = idx + (idx / (k >> 1)) * (k >> 1);
    ixj = i^j;

    affine_point& a_val = values[i];
    affine_point& b_val = values[ixj];
    uint a_key = get_key(powers[i], chunk_num, LARGE_C);
    uint b_key = get_key(powers[ixj], chunk_num, LARGE_C);
    bool order = ((i & k) == 0);

    compare_exchange_points(a_val, b_val, a_key, b_key, order);
}

DEVICE_FUNC __inline__ void bitonic_sort_chunk(affine_point* values, const uint256_g* powers, size_t arr_len, uint chunk_num)
{
    size_t j, k;

    /* Major step */
    for (k = 2; k <= arr_len; k <<= 1)
    {
        /* Minor step */
        for (j = k >> 1; j > 0; j = j >> 1)
        {
            bitonic_sort_step(values, powers, arr_len, j, k, chunk_num);
        }
    }
}

__global__ void bitonic_sort(affine_point* values, const uint256_g* powers, size_t arr_len, uint BLOCKS_PER_BIN)
{
    uint chunk_num = blockIdx.x / BLOCKS_PER_BIN;
    bitonic_sort_chunk(values + arr_len * chunk_num, powers, arr_len, chunk_num);
} 

//NB: here we do also scilently assume that array length is divisible by 64

__global__ void init_bitonic_sort(affine_point* copied_values, const affine_point* points, size_t arr_len)
{
    const uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < arr_len)
    {
        const affine_point* source = &(points[idx]);
       
        #pragma unroll
        for(uint i = 0; i < NUM_OF_CHUNKS; i++)
        {
            affine_point* dest = copied_values + i * arr_len + idx;
            copy_point(source, dest);
        }
        idx += blockDim.x * gridDim.x;
    }
}

void SORT_ARRAY(const affine_point* points, const uint256_g* powers, affine_point* temporary_array, size_t arr_len, uint smCount)
{
    const uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
 
    kernel_geometry init_geometry = find_optimal_geometry(init_bitonic_sort, 0, smCount);
    init_bitonic_sort<<<init_geometry.gridSize, init_geometry.blockSize>>>(temporary_array, points, arr_len);

    kernel_geometry sort_geometry = find_optimal_geometry(bitonic_sort, 0, smCount);
    std::cout << "Sort geometry: " << sort_geometry.blockSize << std::endl;

    //TODO: handle this!
    sort_geometry.blockSize = 256;

    uint NUM_OF_BLOCKS = arr_len * NUM_OF_CHUNKS / (2 * sort_geometry.blockSize);
    uint BLOCKS_PER_BIN = NUM_OF_BLOCKS / NUM_OF_CHUNKS;
    bitonic_sort<<<NUM_OF_BLOCKS, sort_geometry.blockSize>>>(temporary_array, powers, arr_len, BLOCKS_PER_BIN);
}

//second step: counting occurencies
//--------------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC __inline__ void local_density_count(uint256_g* power_arr, size_t arr_start_pos, size_t arr_end_pos, uint chunk_num, size_t* bins)
{
    uint idx = arr_start_pos + threadIdx.x;
    while (idx < arr_end_pos)
    {
        uint key = get_key(power_arr[idx], chunk_num, LARGE_C);
        atomicAdd((uint*)&bins[key], 1);
        idx += blockDim.x;
    }
}

__global__ void init_density_count(size_t* bins)
{
    uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;

    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    while (idx < LARGE_CHUNK_SIZE * NUM_OF_CHUNKS)
    {
        bins[idx] = 0;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void global_density_count(uint256_g* power_arr, size_t* bins, size_t arr_len, uint BLOCKS_PER_BIN)
{   
    uint chunk_num = blockIdx.x / BLOCKS_PER_BIN;
    uint ELEMS_PER_BLOCK = (arr_len + BLOCKS_PER_BIN - 1) / BLOCKS_PER_BIN;
    size_t start_pos = (blockIdx.x % BLOCKS_PER_BIN) * ELEMS_PER_BLOCK;
    size_t end_pos = min(start_pos + ELEMS_PER_BLOCK, arr_len);
    size_t output_pos = LARGE_CHUNK_SIZE * chunk_num;
    
    local_density_count(power_arr, start_pos, end_pos, chunk_num, bins + output_pos);
}

void DENSITY_COUNT(uint256_g* power_arr, size_t* bins, size_t arr_len, uint32_t smCount)
{
    {
        kernel_geometry geometry = find_optimal_geometry(init_density_count, 0, smCount);
        std::cout << geometry.gridSize << ", " <<  geometry.blockSize << std::endl;
        init_density_count<<<geometry.gridSize, geometry.blockSize>>>(bins);
    }
       
    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    
    kernel_geometry geometry = find_optimal_geometry(global_density_count, 0, smCount);
    int minGridSize = geometry.gridSize;
    int blockSize = geometry.blockSize;

    int gridSize = max(minGridSize, NUM_OF_CHUNKS);

    //find the closest power of 2

    uint num = BITS_PER_LIMB - __builtin_clz(gridSize) - 1;
    gridSize = (1 << num);
 
    uint BLOCKS_PER_BIN = gridSize / NUM_OF_CHUNKS;

    global_density_count<<<gridSize, blockSize>>>(power_arr, bins, arr_len, BLOCKS_PER_BIN);
}

//second step: reducing
//------------------------------------------------------------------------------------------------------------------------------------------------

DEVICE_FUNC void block_level_reduce(const affine_point* start, const affine_point* end, ec_point* out)
{
    ec_point acc = point_at_infty();
    
    size_t tid = threadIdx.x;
    const affine_point* cur = start + tid;
	while (cur < end)
	{   
        acc = ECC_MIXED_ADD(acc, *cur);
        cur += blockDim.x;
	}

    acc = blockReduceSum(acc);
    
    if (threadIdx.x == 0)
        *out = acc;
}

//TODO: use many blocks per group!

__global__ void reduce2(affine_point* points, size_t* bins, ec_point* out, size_t arr_len)
{
    size_t i = blockIdx.x / LARGE_CHUNK_SIZE;
    size_t j = blockIdx.x % LARGE_CHUNK_SIZE;
    if ((i == 0) & (j == 0))
        block_level_reduce(points + i * arr_len, points + i * arr_len + bins[i * LARGE_CHUNK_SIZE + j], out + LARGE_CHUNK_SIZE * i + j);
}

void REDUCE_SORTED_ARRAY(affine_point* points, size_t* bins, ec_point* out, size_t arr_len, uint smCount)
{
    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;
    
    kernel_geometry reduce_geometry = find_optimal_geometry(reduce2, 32 * 3 * 8, smCount);
    reduce2<<<NUM_OF_CHUNKS * LARGE_CHUNK_SIZE, reduce_geometry.blockSize>>>(points, bins, out, arr_len); 
}

//other steps: scanning and final shrinking are the same as in the previous algorims

//sort based Pippenger driver
//-----------------------------------------------------------------------------------------------------------------------------------------------------

void sorting_based_Pippenger_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len)
{   
    affine_point* temporary_array1 = nullptr;
    ec_point* temporary_array2 = nullptr;
    size_t* bins = nullptr;

    constexpr uint NUM_OF_CHUNKS = MAX_POWER_BITLEN / LARGE_C;

    std::cout << "array size: " << NUM_OF_CHUNKS * arr_len * sizeof(affine_point) << std::endl;
    
    cudaError_t cudaStatus = cudaMalloc((void **)&temporary_array1, NUM_OF_CHUNKS * arr_len * sizeof(affine_point));
    if (cudaStatus != cudaSuccess)
        std::cout << "Error!" << std::endl;

    cudaStatus = cudaMalloc((void **)&temporary_array2, NUM_OF_CHUNKS * LARGE_CHUNK_SIZE * sizeof(ec_point));
    if (cudaStatus != cudaSuccess)
        std::cout << "Error!" << std::endl;

    cudaStatus = cudaMalloc((void **)&bins, NUM_OF_CHUNKS * LARGE_CHUNK_SIZE * sizeof(size_t));
    if (cudaStatus != cudaSuccess)
        std::cout << "Error!" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

    SORT_ARRAY(point_arr, power_arr, temporary_array1, arr_len, smCount);
    DENSITY_COUNT(power_arr, bins, arr_len, smCount);
    REDUCE_SORTED_ARRAY(temporary_array1, bins, temporary_array2, arr_len, smCount);

    //SCAN_ARRAY(temporary_array2, (ec_point*)temporary_array1, smCount);
    
    //TODO: there is no need for additional array allocation!
    
    //REDUCE_ARRAY(temporary_array1, out_temp_arr, locks, smCount);
    //cudaMemcpy(out, out_temp_arr, 65536 * sizeof(ec_point), cudaMemcpyDeviceToDevice);
    
    //final_shrinking<<<1, NUM_OF_CHUNKS>>>(out_temp_arr, out, 1);
    
    cudaFree(temporary_array1);
    cudaFree(temporary_array2);
    cudaFree(bins);
}
