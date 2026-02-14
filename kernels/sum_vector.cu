#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>


__global__ void sum_vector(int *a, int *output, int N){
    __shared__ int sdata[1024];
    int stride = blockDim.x/2;
    int block_offset_x = blockDim.x * blockIdx.x;
    int index = block_offset_x + threadIdx.x;
    sdata[threadIdx.x] = 0;
    if(index<N)
    sdata[threadIdx.x] = a[index];
    __syncthreads();
    for(int step = stride; step>0; step/=2){
        if(threadIdx.x < step){
            sdata[threadIdx.x] += sdata[threadIdx.x+step];
        }
        __syncthreads();
    }
    if(threadIdx.x==0)
        atomicAdd(output, sdata[0]);
}


float test_sum_vector(int N=100){
    int *a, *d_a, *output, *d_o;
    int bytes=N * sizeof(int);
    cudaMallocHost((void**)&a, bytes);
    cudaMalloc((void **)&d_a, bytes);
    for(int i=0; i<N; i+=1)
        a[i] = i+1;
    cudaEvent_t start, end;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    int block_size = 1024;
    int blocks = (block_size + N -1)/block_size;
    cudaMallocHost((void **)&output, sizeof(int));
    output[0] = 0;
    cudaMalloc((void **)&d_o, sizeof(int));
    cudaMemcpy(d_o, output, 1*(sizeof(int)), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    sum_vector<<<blocks, block_size>>>(d_a, d_o, N);
    cudaDeviceSynchronize();
    cudaEventRecord(end, 0);
    cudaMemcpy(output, d_o, sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Final Sum:- %d ", *output);
    printf("\n");
    return elapsed_time;
}

int main(){
    test_sum_vector(); //warmup
    printf("total Time:- %f\n", test_sum_vector(1024 * 10));
}