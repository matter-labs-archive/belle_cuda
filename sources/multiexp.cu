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
        b[i].x = __shfl_down_sync(a[i].x, offset, width);
        b[i].y = __shfl_down_sync(a[i].y, offset, width);
        b[i].z = __shfl_down_sync(a[i].z, offset, width);
        b[i].w = __shfl_down_sync(a[i].w, offset, width);
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

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_atomics, 0, 4 * N * 3 * WARP_SIZE);
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

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_multiexp_kernel_block_level_recursion, 0, SHARED_MEMORY_USED);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, naive_multiexp_kernel_block_level_recursion, blockSize, SHARED_MEMORY_USED);
    maxExpGridSize = maxActiveBlocks * smCount;
    ExpBlockSize = blockSize;

    //the same routine for reduction phase

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, naive_kernel_block_level_reduction, 0, SHARED_MEMORY_USED);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, naive_kernel_block_level_reduction, blockSize, SHARED_MEMORY_USED);
    maxReductionGridSize = maxActiveBlocks * smCount;
    ReductionBlockSize = blockSize;

    //do the first stage:

    realGridSize = (arr_len + ExpBlockSize - 1) / ExpBlockSize;
    realGridSize = min(realGridSize, maxExpGridSize);

	std::cout << "Real grid size: " << realGridSize << ",  blockSize: " << ExpBlockSize << std::endl;
	naive_multiexp_kernel_block_level_recursion<<<realGridSize, ExpBlockSize>>>(point_arr, power_arr, out_arr, arr_len);

    //NB: we also use input array as a source for temporary output, so that it's content will be destroyed

    arr_len = realGridSize;
    ec_point* temp_input_arr = out_arr;
    ec_point* temp_output_arr = reinterpret_cast<ec_point*>(point_arr);
    unsigned iter_count = 0;

    while (arr_len > 1)
    {
        cudaDeviceSynchronize();
        std::cout << "iter " << ++iter_count << ", real grid size: " << realGridSize << ",  blockSize: " << ReductionBlockSize << std::endl;
        realGridSize = (arr_len + ReductionBlockSize - 1) / ReductionBlockSize;
        realGridSize = min(realGridSize, maxReductionGridSize);

        naive_kernel_block_level_reduction<<<realGridSize, ReductionBlockSize>>>(temp_input_arr, temp_output_arr, arr_len);
        arr_len = realGridSize;

        //swap arrays
        ec_point* swapper = temp_input_arr;
        temp_input_arr = temp_output_arr;
        temp_output_arr = swapper;

    }

    //copy output to the correct array! (but we are just moving pointers :)
    out_arr = temp_output_arr;
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

    while(__any_sync(peers, peers))
    {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl() only works if both threads participate, so we always do.
        ec_point temp;
        __shfl(pt, temp, peers, next - 1);

        // only add if there was anything to add
        if (next)
        {
            pt = ECC_ADD(pt, temp);
        }

        // all lanes with their least significant index bit set are done
        uint32_t done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~ __ballot_sync(peers, done);

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


DEVICE_FUNC __inline__ uint get_key(const uint256_g& val, uint chunk_num)
{
    uint bit_pos = chunk_num * SMALL_CHUNK_SIZE;
    uint limb_idx = bit_pos / BITS_PER_LIMB;
    uint offset = bit_pos % BITS_PER_LIMB;

    uint low_part = val.n[limb_idx];
    uint high_part = (limb_idx < N - 1 ? val.n[limb_idx + 1] : 0);

    uint res = __funnelshift_r(low_part, high_part, offset);
    res &= (1 << SMALL_C) - 1;

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
        uint key = get_key(power_arr[idx], chunk_num);

        uint peers = get_peers(key);
        uint leader = reduce_peers(peers, pt);
        
        if (lane == leader)
        {
            uint real_key = REVERSE_INDEX(key, SMALL_CHUNK_SIZE);

            bool leaveLoop = false;
            while (!leaveLoop)
            {
                if ( bins[real_key].lock.try_lock())
                {
                    //critical section
                    bins[real_key].pt = ECC_ADD(pt, bins[real_key].pt);
                    leaveLoop = true;
                    bins[real_key].lock.unlock();
                    
                }
                __threadfence_block();
                //printf("here\n");
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
    size_t start_pos = blockIdx.x * ELEMS_PER_BLOCK;
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
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif


//NB: arr_len should be a power of two
//TBD: implement comflict free offsets

#define PIPPENGER_BLOCK_SIZE 256

__global__ void scan_and_reduce(const ec_point* global_in_arr, ec_point* out)
{
    // allocated on invocation
    __shared__ ec_point temp[PIPPENGER_BLOCK_SIZE * 2];

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
            temp[tid] = temp[tid + d];
        __syncthreads();
    }

    if (tid == 0)
        out[blockIdx.x * SMALL_CHUNK_SIZE] = temp[0];
}


//Pippenger: basic version - simple, yet powerful. The same version of Pippenger algorithm is implemented in libff and Bellman


void Pippenger_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
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

    gridSize = (arr_len + PIPPENGER_BLOCK_SIZE - 1) / PIPPENGER_BLOCK_SIZE;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, device_level_histo, PIPPENGER_BLOCK_SIZE, SHARED_MEMORY_USED);
    gridSize = min(maxActiveBlocks * smCount, gridSize);
    gridSize = max(gridSize, NUM_OF_CHUNKS);

    //find the closest power of 2

    uint num = BITS_PER_LIMB - __builtin_clz(gridSize) - 1;
    gridSize = (1 << num);

    //-----------------------------------------------------------------------------------------------------------------
    //calculate kernel run parameteres
 
    uint BLOCKS_PER_BIN = gridSize / NUM_OF_CHUNKS;
    //uint ELEMS_PER_BLOCK = (arr_len + BLOCKS_PER_BIN - 1) / BLOCKS_PER_BIN;

    //allocate memory for temporary array of needed

    ec_point* histo_arr = nullptr;
    size_t HISTO_ELEMS_COUNT = SMALL_CHUNK_SIZE * gridSize;
    cudaMalloc((void **)&histo_arr, HISTO_ELEMS_COUNT * sizeof(ec_point));
    
    //run kernels - one after after another:
    //1) collect local block-level histograms
    //2) shrink all local histograms to a larger one
    //3) perform block level scan and reduce on shrinked histogram

    gridSize = NUM_OF_CHUNKS;
    device_level_histo<<<gridSize, PIPPENGER_BLOCK_SIZE>>>(point_arr, power_arr, histo_arr, arr_len, BLOCKS_PER_BIN);
   
    cudaDeviceSynchronize();

    shrink_histo<<<NUM_OF_CHUNKS, PIPPENGER_BLOCK_SIZE>>>(histo_arr, out_arr, BLOCKS_PER_BIN);
    cudaDeviceSynchronize();

    scan_and_reduce<<<NUM_OF_CHUNKS, PIPPENGER_BLOCK_SIZE>>>(out_arr, out_arr);

    cudaFree(histo_arr);
}


//Some experiements with CUB library
//----------------------------------------------------------------------------------------------------------------------------------------------

//Do not compile: may be I should ask on stackoverflow?

#ifdef OLOLO_TROLOLO

#include <cub/cub.cuh>   

struct ec_point_adder_t
{
    DEVICE_FUNC __forceinline__
    ec_point operator()(const ec_point& a, const ec_point& b) const
    {
        return ECC_ADD(a, b);
    }
};

__global__ void exp_componentwise(const affine_point* point_arr, const uint256_g* power_arr, ec_point* out, size_t arr_len)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < arr_len)
	{   
        out[tid] = ECC_EXP(point_arr[tid], power_arr[tid]);
        tid += blockDim.x * gridDim.x;
	} 
}

void CUB_reduce_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out_arr, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;
    int maxActiveBlocks;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

  	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, exp_componentwise, 0, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, exp_componentwise, blockSize, 0);

    realGridSize = (arr_len + blockSize - 1) / blockSize;
    realGridSize = min(realGridSize, maxActiveBlocks * smCount);
      
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    ec_point_adder_t ec_point_adder;

     ec_point infty = { 
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000001},
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}
    };


    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, out_arr, out_arr, arr_len, ec_point_adder, infty);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run reduction
    //cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,  out_arr, out_arr, arr_len, ec_point_adder, point_at_infty());

    cudaFree(d_temp_storage);
}

#endif



