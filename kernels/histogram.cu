#include<iostream>
#include<cuda.h>

__global__ void histogram(int *a, int *bins, int N, int B, int min_array, int width){
    int bx = blockIdx.x;
    int block_offset = bx * blockDim.x;
    int tx = threadIdx.x;
    int index = block_offset + tx;
    int bin_pos = -1;
    if(index<N)
        bin_pos = (a[index] - min_array)/width;
    if(bin_pos >=0 && bin_pos < B)
        atomicAdd(bins + bin_pos, 1);
}

__global__ void histogram_optimal(int *a, int *bins, int N, int B, int min_array, int width){
    extern __shared__ int s_bins[];
    for(int i=threadIdx.x; i<B; i+=blockDim.x)
        s_bins[i] = 0;
    __syncthreads();
    int gx = gridDim.x;
    int bx = blockIdx.x;
    int block_offset = bx * blockDim.x;
    int tx = threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i = 0;
    int index = block_offset + tx;
    for(int i=index; i< N; i+= stride){
        int val = a[i];
        int bin_pos = (val-min_array)/width;
        if(bin_pos<B)
            atomicAdd(s_bins + bin_pos, 1);
    }
    __syncthreads();
    for(int i=tx; i<B; i+=blockDim.x)
        atomicAdd(bins+i, s_bins[i]);
}

void print(int *a, int N){
    printf("\n");
    for(int i=0; i<N; i+=1){
        printf("%d ", a[i]);
    }
    printf("\n");
}


void test_histogram_naive(int N, int B){
    int *a, *d_a, *bins, *d_bins;
    int bytes = N * sizeof(int);
    int bin_bytes = B * sizeof(int);
    cudaMallocHost((void **)&a, bytes);
    cudaMallocHost((void **)&bins, bin_bytes);
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_bins, bin_bytes);
    cudaMemset(d_bins, 0, bin_bytes);
    for(int i=0; i<N; i+=1)
        a[i] = i+2;
    int min_array = 2;
    int max_array = N + 2;
    int range = max_array - min_array;
    int width = range/B;
    if(range % B != 0)
        width += 1;
    int max_threads = 1024;
    int blocks = (N + max_threads-1)/max_threads;
    printf("Original Array:- \n");
    print(a, N);
    printf("Width:- %d\n", width);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    histogram<<<blocks, max_threads>>>(d_a, d_bins, N, B, min_array, width);
    cudaMemcpy(bins, d_bins, bin_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    print(bins, B);
    cudaFreeHost(a);
    cudaFree(d_a);
    cudaFree(d_bins);
    cudaFreeHost(bins);
}


void test_histogram_optimal(int N, int B){
    int *a, *d_a, *bins, *d_bins;
    int bytes = N * sizeof(int);
    int bin_bytes = B * sizeof(int);
    cudaMallocHost((void **)&a, bytes);
    cudaMallocHost((void **)&bins, bin_bytes);
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_bins, bin_bytes);
    cudaMemset(d_bins, 0, bin_bytes);
    for(int i=0; i<N; i+=1)
        a[i] = i+2;
    int min_array = 2;
    int max_array = N + 2;
    int range = max_array - min_array;
    int width = range/B;
    if(range % B != 0)
        width += 1;
    int max_threads = 1024;
    int blocks = 4;
    //printf("Original Array:- \n");
    //print(a, N);
    printf("Width:- %d\n", width);
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    histogram_optimal<<<blocks, max_threads, bin_bytes>>>(d_a, d_bins, N, B, min_array, width);
    cudaMemcpy(bins, d_bins, bin_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    print(bins, B);
    cudaFreeHost(a);
    cudaFree(d_a);
    cudaFree(d_bins);
    cudaFreeHost(bins);
}


int main(){
    test_histogram_optimal(300000, 25);
}