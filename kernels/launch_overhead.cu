#include<iostream>
#include<cuda.h>

__global__ void do_it(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}

int main(){
    int *a, *d_a;
    cudaEvent_t start, end;
    float elapsed_time;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
    for(int i=0; i<10000; i+=1)
        do_it<<<1,1>>>();
    cudaEventRecord(end, 0);
    cudaDeviceSynchronize();
    printf("Type,Total_Time_ms,Avg_Time_Per_Launch_us\n");
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Loop_10k_Launches,%f,%f\n", elapsed_time, elapsed_time/10);

    cudaEventRecord(start, 0);
    do_it<<<10000, 1>>>();
    cudaEventRecord(end, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time, start, end);
    printf("Single_Batch_Launch,%f,%f\n", elapsed_time, elapsed_time/10);
    return 0;
}