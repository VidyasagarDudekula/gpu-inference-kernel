#include<iostream>
#include<nvbench/nvbench.cuh>

__global__ void add_it(int *a, int *b, int *c, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index<N)
    c[index] = a[index] + b[index];
}

void vector_benchmark(nvbench::state &state){
    int *a, *b, *c, *d_a, *d_b, *d_c;
    int N = state.get_int64("Elements");
    int bytes = N * sizeof(int);
    int max_threads = 1024;
    int blocks = (N+max_threads-1)/max_threads;
    cudaMallocHost((void **)&a, bytes);
    cudaMallocHost((void **)&b, bytes);
    cudaMallocHost((void **)&c, bytes);
    for(int i=0; i<N; i+=1){
        a[i] = i;
        b[i] = i+N;
    }
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch){
        add_it<<<blocks, max_threads>>>(d_a, d_b, d_c, N);
    });
    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    // for(int i=0; i<N; i+=1){
    //     printf("%d + %d = %d\n", a[i], b[i], c[i]);
    // }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
}

NVBENCH_BENCH(vector_benchmark).add_int64_axis("Elements", {1024, 65536, 1048576, 10485760});

NVBENCH_MAIN