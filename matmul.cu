/**
 * Matmul kernels
 *  - Naive
 *  - Tiled
 *  - Tiled + unroll
 */

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


template<int TILE_SIZE>
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
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // ceil(K / TILE_SIZE)
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    float acc = 0.f;

    for (int i = 0; i < numTiles; i++){
        // Load a single value into SMEM tile
        // thread (tx, ty) is responsible for loading 
        tileA[ty][tx] = A[row * K + col];

        // Wait for all other threads in block to have put the GMEM cell into SMEM tile
        // There are TILE_SIZE**2 threads
        __syncthreads();

        // tileA, tileB are now filled and we can compute the dot product
        // Whether or not to unroll the compile-known constant TILE_SIZE
        // #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tileA[ty][k] * tileB[k][tx];
        }
    
        // wait for all threads in block to have finished computing.
        // So we dont start overwriting memory in the next loop iteration
        __syncthreads();
    }

    // epilogue: write back to GMEM
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
