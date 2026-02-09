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

__global__ void do_it(int *a, int *b, int *c, int *indices, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < N){
        c[index] = a[indices[index]] + b[indices[index]];
    }
}

void get_indices_random(int *indices, int N){
    std::random_device rd;
    std::mt19937 gen(rd());
    int min=0, max=N-1;
    std::uniform_int_distribution<> distrib(min, max);
    for(int i=0; i<N; i+=1){
        int random_num = distrib(gen);
        indices[i] = random_num;
    }
}

void get_inidices_strides(int *indices, int N, int k){
    for(int i=0; i<N; i+=1){
        indices[i] = ((long long)i * k)%N;
    }
}

int main(){
    int N = 1024 * 100000;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_threads = prop.maxThreadsPerBlock;
    int blocks = (N + max_threads -1)/max_threads;
    int groups = 4;
    cudaStream_t streams[groups];
    int group_size = N/groups;

    int min =0;
    int max = N-1;
    std::uniform_int_distribution<> ditrib(min, max);
    int *a, *b, *c, *indices, *d_a, *d_b, *d_c, *d_i;
    int bytes = N * sizeof(int);
    int group_bytes = group_size * sizeof(int);
    CUDA_CHECK(cudaMallocHost((void **)&a, bytes));
    CUDA_CHECK(cudaMallocHost((void **)&b, bytes));
    CUDA_CHECK(cudaMallocHost((void **)&c, bytes));
    CUDA_CHECK(cudaMallocHost((void **)&indices, bytes));

    CUDA_CHECK(cudaMalloc((void **)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_c, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_i, bytes));

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    for(int i=0; i<N; i+=1){
        a[i] = i;
        b[i] = N-i-25;
    }
    //warmup
    get_inidices_strides(indices, N, 1);
    CUDA_CHECK(cudaMemcpy(d_i, indices, bytes, cudaMemcpyHostToDevice));
    do_it<<<blocks, max_threads>>>(d_a, d_b, d_c, d_i, N);

    printf("Random:-\n");
    get_indices_random(indices, N);
    CUDA_CHECK(cudaMemcpy(d_i, indices, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start, 0));
    do_it<<<blocks, max_threads>>>(d_a, d_b, d_c, d_i, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, end));
    printf("total time:- %f ms\n", elapsed_time);

    for(int k=1; k<129; k*=2){
        get_inidices_strides(indices, N, k);
        printf("Stride k=%d\n", k);
        CUDA_CHECK(cudaMemcpy(d_i, indices, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start, 0));
        do_it<<<blocks, max_threads>>>(d_a, d_b, d_c, d_i, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(end, 0));
        CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed_time;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, end));
        printf("total time:- %f ms\n", elapsed_time);
    }
}