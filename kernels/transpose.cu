#include<iostream>
#include<cuda.h>
#include<random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err__ = (call);                                          \
    if (err__ != cudaSuccess) {                                          \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__     \
                  << " : " << cudaGetErrorString(err__) << std::endl;    \
        std::exit(1);                                                    \
    }                                                                    \
} while (0)

__global__ void transpose(int *a, int *b, int N, int M){
    __shared__ int tile[16][17];

    int in_row = blockIdx.y * 16 + threadIdx.y;
    int in_col = blockIdx.x * 16 + threadIdx.x;

    int x = threadIdx.x;
    int y = threadIdx.y;

    int val = 0;
    if (in_row < N && in_col < M)
        val = a[in_row * M + in_col];

    tile[y][x] = val;

    __syncthreads();

    int out_row = blockIdx.x * 16 + threadIdx.y;
    int out_col = blockIdx.y * 16 + threadIdx.x;

    if (out_row < M && out_col < N)
        b[out_row * N + out_col] = tile[x][y];
}

void print(int *a, int N, int M){
    for(int i=0; i<N*M; i+=1){
        if(i%M==0)
            printf("\n");
        printf("%d ", a[i]);
    }
    printf("\n\n");
}

int main(){
    int *a, *b, N, M, *d_a, *d_b;
    N = 11;
    M = 9;
    int bytes = M * N * sizeof(int);
    CUDA_CHECK(cudaMallocHost((void **)&a, bytes));
    CUDA_CHECK(cudaMallocHost((void **)&b, bytes));

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    int total = N * M;
    for(int i=0; i<total; i+=1){
        a[i] = i+1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_threads = prop.maxThreadsPerBlock;
    dim3 block(16, 16);
    dim3 grid((M + block.x-1)/block.x, (N + block.y - 1)/block.y);

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    transpose<<<grid, block>>>(d_a, d_b, N, M);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost));
    print(a, N, M);
    print(b, M, N);
}

