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


__global__ void transpose_optimal(int *a, int *b, int N, int M){
    __shared__ int tile[32][33];
    int block_offset_x = blockIdx.x * blockDim.x;
    int block_offset_y = blockIdx.y * blockDim.y;
    int global_x = threadIdx.x + block_offset_x;
    int global_y = threadIdx.y + block_offset_y;
    int index_a = global_y * M + global_x;
    tile[threadIdx.y][threadIdx.x] = 0;
    if(global_y<N && global_x<M)
        tile[threadIdx.y][threadIdx.x] = a[index_a];
    __syncthreads();
    int global_write_x = block_offset_y + threadIdx.x;
    int global_write_y = block_offset_x + threadIdx.y;
    int index_b = global_write_y * N + global_write_x;
    if(global_write_x<N && global_write_y < M)
    b[index_b] = tile[threadIdx.x][threadIdx.y];
}

__global__ void naive_2d_transpose(int *a, int *b, int N, int M){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.x + threadIdx.y;

    if(row<N && col<M){
        int index_a = row*M + col;
        int index_b = col*N + row;
        // printf("row:- %d, col:- %d, index_a:- %d, index_b:- %d\n", row, col, index_a, index_b);
        b[index_b] = a[index_a];
    }
}


__global__ void naive_1d_transpose(int *a, int *b, int N, int M){
    int index_a = blockIdx.x * blockDim.x + threadIdx.x;
    int row = index_a/M;
    int col = index_a%M;

    if(row<N && col<M){
        int index_b = col * N + row;
        b[index_b] = a[index_a];
    }
}


void print(int *a, int N, int M){
    for(int i=0; i<N*M; i+=1){
        printf("%d ", a[i]);
        if((i+1)%M==0)
            printf("\n");
    }
}

int main(){
    int *a, *b, *d_a, *d_b, *d_junk, *d_b2, *d_b3, N, M;
    N = 4096; M = 4096;
    float elapsed_time;
    int bytes = N * M * sizeof(int);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_threads = prop.maxThreadsPerBlock;
    cudaMallocHost((void **)&a, bytes);
    cudaMallocHost((void **)&b, bytes);
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_b2, bytes);
    cudaMalloc((void **)&d_b3, bytes);
    cudaMalloc((void **)&d_junk, bytes);

    int total = N * M;
    int n_blocks = (total + max_threads-1)/max_threads;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    for(int i=1; i<=N*M; i+=1)
        a[i-1] = i;
    // print(a, M, N);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    naive_1d_transpose<<<n_blocks, max_threads>>>(d_a, d_junk, N, M);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    naive_1d_transpose<<<n_blocks, max_threads>>>(d_a, d_b, N, M);
    cudaEventRecord(end, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsed_time, start, end);
    // print(b, M, N);
    printf("time it took for 1d array:- %f\n", elapsed_time);

    dim3 blocks(16, 16);
    dim3 gird((blocks.x + M -1)/blocks.x, (blocks.y + N-1)/blocks.y);

    cudaEventRecord(start, 0);
    naive_2d_transpose<<<gird, blocks>>>(d_a, d_b2, N, M);
    cudaEventRecord(end, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(b, d_b2, bytes, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsed_time, start, end);
    // print(b, M, N);
    printf("time it took for 2d array:- %f\n", elapsed_time);

    cudaEventRecord(start, 0);
    transpose_optimal<<<gird, blocks>>>(d_a, d_b3, N, M);
    cudaEventRecord(end, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(b, d_b3, bytes, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsed_time, start, end);
    // print(b, M, N);
    printf("time it took for 2d array:- %f\n", elapsed_time);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFree(d_junk);
    cudaFree(d_a);
    cudaFree(d_b2);
    cudaFree(d_b);
    cudaFree(d_b3);
}
