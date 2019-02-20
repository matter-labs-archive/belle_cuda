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

#define SMALL_CHUNK_SIZE 256
#define LARGE_CHUNK_SIZE 256

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

    uint res = __funnelshift_r(low, high, offset);
    res &= (1 << SMALL_CHUNK_SIZE) - 1;

    return res;   
}

DEVICE_FUNC __inline__ ec_point from_affine_point(const affine_point& pt)
{
    return ec_point{pt.x, pt.y, BASE_FIELD_R};
}


DEVICE_FUNC __inline__ void block_level_histo(affine_point* pt_arr, uint256_g* power_arr, 
    size_t arr_start_pos, size_t arr_end_pos, uint chunk_num, ec_point* out_histo_arr)
{
    //we exclude the bin corresponding to value 0
    
    static __shared__ Bin bins[SMALL_CHUNK_SIZE]; 
    uint lane = threadIdx.x % warpSize;
    uint wid = threadIdx.x / warpSize;
    
    //first we need to init all bins

    size_t idx = threadIdx.x;
    while (idx < SMALL_CHUNK_SIZE)
    {
        bins[idx].lock.init()
        bins[idx].pt = point_at_infty();
        idx += blockDim.x;
    }

    __syncthreads();

    idx = arr_start_pos + threadId.x;
    while (idx < arr_end_pos)
    {
        ec_point pt = from_affine_point(pt_arr[idx]);
        uint key = get_key(power_arr[idx], chunk_num);

        if (key > 0)
        {
            uint peers = get_peers(key);
            uint leader = reduce_peers(peers, pt);

            if (wid == leader)
            {
                bins[key].lock.lock();
                bins[key].pt = ECC_ADD(pt, bins[key].pt);
                bins[key].lock.unlock();
            }
        }

        idx += blockDim.x;
    }

    __syncthreads();

    size_t idx = threadIdx.x;
    while (idx < SMALL_CHUNK_SIZE)
    {
        out[idx] = bins[idx].pt;
        idx += blockDim.x;
    }
}

//Another important part of Pippenger algorithm is the evaluation of Rolling sum - we use the combination of scan + reduction primitives
//We do also try to avoid bank conflicts (although it sounds like a sort of some weird magic)

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

//NB: arr_len should be a power of two

__global__ void block_prescan(const ec_point* in_arr, ec_point* out_arr, ec_point* block_sums, size_t arr_len)
{
    __shared__ ec_point temp[];
    uint tid = threadIdx.x;
    uint offset = 1;

    uint ai = tid;
    uint bi = thid + (arr_len / 2);
    uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint bankOffsetB = CONFLICT_FREE_OFFSET(ai);
    temp[ai + bankOffsetA] = in_arr[ai];
    temp[bi + bankOffsetB] = in_arr[bi]; 

    for (int d = arr_len >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();
        if (thd < d)
        {
            uint ai = offset * (2 * tid+1)-1;
            uint bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 

            temp[bi] = ECC_ADD(temp[ai], temp[bi]);
        }
        offset *= 2;
    }
    
    if (tid==0)
    {
        temp[arr_len - 1 + CONFLICT_FREE_OFFSET(arr_len - 1)] = 0;
    }

    // traverse down tree & build scan
    for (uint d = 1; d < arr_len; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (thd < d)
        {
            uint ai = offset * (2 * tid+1)-1;
            int bi = offset*(2*thid+2)-1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi); 
            ec_point t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = ECC_ADD(t, temp[bi]);
        }
    }
    
    __syncthreads();
   
    out_arr[ai] = out_arr[ai + bankOffsetA];
    out_arr[bi] = temp[bi + bankOffsetB];

    if (tid==0)
    {

    }
}

//scan of large arrays :()


//third step - reduction of rolling sum

//Pippenger: basic version - simple, yet powerful. The same version of Pippenger algorithm is implemented in libff and Bellman

void Pippenger_driver(affine_point* point_arr, uint256_g* power_arr, ec_point* out, size_t arr_len)
{
    int blockSize;
  	int minGridSize;
  	int realGridSize;
    int maxActiveBlocks;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
	uint32_t smCount = prop.multiProcessorCount;

    //first - stage: splitting into bins




//Bitonic Sort
//------------------------------------------------------------------------------------------------------------------------------------------
//TODO: may be better and simpler to use CUB library? (it contains primitives for sorting)

__global__ void bitonic_sort_step(ec_point* values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}






