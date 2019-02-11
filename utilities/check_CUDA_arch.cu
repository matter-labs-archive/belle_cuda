#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
    cudaDeviceProp dP;

    int rc = cudaGetDeviceProperties(&dP, 0);
    if(rc != cudaSuccess)
    {
        cudaError_t error = cudaGetLastError();
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return rc; /* Failure */
    }
    else
    {
        printf("%d%d", dP.major, dP.minor);
        return 0; /* Success */
    }
}