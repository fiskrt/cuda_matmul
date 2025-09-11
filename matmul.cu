


__global__ void matmul_kernel(
    float *C, const float *A, const float *B,
    int M, int N, int K
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float value = 0.0f;
        for (int e = 0; e < K; ++e)
        {
            value += A[row * K + e] * B[e * N + col];
        }
        // prologue: write back to GMEM
        C[row * N + col] = value;
    }
}


#define TILE_SIZE 16
__global__ void matmul_kernel_tiled(
    float *C, const float *A, const float *B,
    int M, int N, int K
)
{
    // Q1: why do we even benefit of loading to SMEM?
    // Cuz one thread loads one element, but many others use it

    // We can have 2d arrays here since we know sizes at compile time.
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Global row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int numTiles = 1;
    
}