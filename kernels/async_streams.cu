#include<iostream>
#include<cuda.h>

__global__ void do_it(int *a, int *b, int *c, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index<N)
        c[index] = a[index] + b[index];
}

int main(){
    int N = 1024*100000;
    int groups = 4;
    int group_size = N/groups;
    int bytes = N * sizeof(int);
    int *a, *d_a, *b, *d_b, *c, *d_c;
    cudaEvent_t start, end;
    float elapsed_time;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaStream_t streams[groups];
    cudaMallocHost((void **)&a, bytes);
    cudaMallocHost((void **)&b, bytes);
    cudaMallocHost((void **)&c, bytes);
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);
    for(int i=0; i<N;i +=1)
    {
        a[i] = i;
        b[i] = N-i-25;
    }
    // 0, 4, 8, 12, 16, 20, 24, 28, 32, 26, 40 --> memory location with bytes difference.
    // group-size = 3
    // a[0*group_size] = 0
    // a[1*group_size] = 12
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    int max_threads = prop.maxThreadsPerBlock;
    int blocks = (group_size+max_threads-1)/max_threads;
    for(int i=0; i<groups; i+=1)
        cudaStreamCreate(&streams[i]);
    cudaEventRecord(start, 0);
    int group_bytes = group_size * sizeof(int);
    for(int i=0; i<groups; i+=1){
        cudaMemcpyAsync(d_a + (i*group_size), a + (i*group_size), group_bytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b + (i*group_size), b + (i*group_size), group_bytes, cudaMemcpyHostToDevice, streams[i]);
    }
    for(int i=0; i<groups; i+=1){
        do_it<<<blocks, max_threads, 0, streams[i]>>>(d_a+ (i*group_size), d_b+ (i*group_size), d_c+ (i*group_size), group_size);
    }
    for(int i=0; i<groups; i+=1){
        cudaMemcpyAsync(c+(i*group_size), d_c + (i*group_size), group_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Total Time:- %f\n", elapsed_time);
    // for(int i=0; i<N; i++){
    //     printf("%d ", c[i]);
    // }
    // printf("\n");
    return 0;
}