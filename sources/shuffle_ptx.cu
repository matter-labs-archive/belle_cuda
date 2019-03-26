__constant__ unsigned A[32];

__global__ void I_wanna_understand_shuffles()
{
    unsigned c = A[threadIdx.x];
    unsigned b = __shfl_down_sync(0xffffffff, c, 1, 8);
    
}

