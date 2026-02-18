#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>


__global__ void prefix_sum(int *a, int *d_o, int N, int prev_sum){
    __shared__ int tile[1024];
    int block_offset_x = blockIdx.x * blockDim.x;
    int tx = (int)threadIdx.x;
    int index = block_offset_x + tx;
    if(index<N)
    tile[tx] = a[index];
    if(tx==0)
        tile[tx] += prev_sum;
    __syncthreads();
    for(int step=1; step<=512; step*=2){
        int val = 0;
        if(index<N)
        val = tile[tx];
        if(tx - step >=0 && tx - step <N){
            val += tile[tx-step];
        }
        __syncthreads();
        if(index<N){
            tile[tx] = val;
        }
        __syncthreads();
    }
    if(index<N)
        d_o[index] = tile[tx];
}


void test_prefix_sum(int N=100){
    int *a, *b, *d_a, *d_b, *output, *d_o;
    int total = N;
    int bytes = total * sizeof(int);
    cudaMallocHost((void **)&a, bytes);
    cudaMallocHost((void **)&output, bytes);
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_o, bytes);
    cudaMemset(d_o, 0, bytes);
    for(int i=0; i<total; i+=1){
        a[i] = i+1;
    }
    int max_threads = 1024;
    int blocks = (N+max_threads - 1)/max_threads;
    int prev_sum = 0;
    for(int i=0; i<blocks; i+=1){
        int threads_offset = (i * 1024);
        int rem_element = N - threads_offset;
        int chunk = std::min(rem_element, max_threads);
        int last_element = chunk-1;
        cudaMemcpy(d_a + threads_offset, a + threads_offset, sizeof(int)*chunk, cudaMemcpyHostToDevice);
        prefix_sum<<<1, max_threads>>>(d_a + threads_offset, d_o + threads_offset, rem_element, prev_sum);
        cudaDeviceSynchronize();
        cudaMemcpy(output + threads_offset, d_o + threads_offset, sizeof(int)*chunk, cudaMemcpyDeviceToHost);
        prev_sum = output[threads_offset + last_element];
    }
    int actaul_val = (N * (N+1))/2;
    printf("last index:- %d\nActual:- %d\n", output[N-1], actaul_val);
}

int main(){
    test_prefix_sum(27293);
}