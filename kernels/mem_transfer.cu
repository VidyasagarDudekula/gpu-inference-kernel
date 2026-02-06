#include<iostream>
#include<cuda.h>

struct customeDict{
    int n_value;
    bool is_pinned;
    float total_time_took;
    float htod_time;
    float dtoh_time;
    float compute_time;
};

__global__ void square_it(int *a){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    a[index] = a[index] * a[index];
}

int main(){
    int *a, *d_a;
    int N_list[] = {100, 1024, 1024*10, 1024*100, 1024*1000, 1024*10000, 1024*100000};
    bool pinned_list[] = {true, false};
    int total_values = int(std::size(N_list)) * int(std::size(pinned_list));
    customeDict *data_list = new customeDict[total_values];
    int counter = 0;
    for(auto N: N_list){
        int bytes = N  * sizeof(int);
        for(auto isPinned: pinned_list){
            cudaEvent_t start, end, HTOD_start, HTOD_end, DTOH_start, DTOH_end, compute_start, compute_end;
            float elapsed_time, elapsed_time_HTOD, elapsted_time_DTOH, elapsted_time_compute;
            int max_threads = 1024;
            int blocks = (N + max_threads-1)/max_threads;
            if(isPinned)
                cudaMallocHost((void **)&a, bytes);
            else
                a = new int[N];
            cudaMalloc((void **)&d_a, bytes);

            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventCreate(&HTOD_start);
            cudaEventCreate(&HTOD_end);
            cudaEventCreate(&DTOH_start);
            cudaEventCreate(&DTOH_end);
            cudaEventCreate(&compute_start);
            cudaEventCreate(&compute_end);
            cudaEventRecord(start, 0);
            cudaEventRecord(HTOD_start, 0);
            cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(HTOD_end, 0);
            cudaEventRecord(compute_start, 0);
            square_it<<<blocks, max_threads>>>(d_a);
            cudaEventRecord(compute_end, 0);
            cudaEventRecord(DTOH_start, 0);
            cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(DTOH_end, 0);
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed_time, start, end);
            cudaEventElapsedTime(&elapsed_time_HTOD, HTOD_start, HTOD_end);
            cudaEventElapsedTime(&elapsted_time_compute, compute_start, compute_end);
            cudaEventElapsedTime(&elapsted_time_DTOH, DTOH_start, DTOH_end);
            customeDict temp;
            temp.n_value = N;
            temp.is_pinned = isPinned;
            temp.total_time_took = elapsed_time;
            temp.htod_time = elapsed_time_HTOD;
            temp.dtoh_time = elapsted_time_DTOH;
            temp.compute_time = elapsted_time_compute;
            data_list[counter] = temp;
            counter += 1;
            cudaFree(d_a);
            if(isPinned)
                cudaFreeHost(a);
            else
                delete[] a;
        }
    }
    for(int i=0; i<total_values; i+=1){
        customeDict temp = data_list[i];
        printf("| N:- %d | isPinned:- %d | elapsed_total_time:- %f | HTOD_time:- %f | DTOH_time:- %f | compute_time:- %f |\n", temp.n_value, temp.is_pinned, temp.total_time_took, temp.htod_time, temp.dtoh_time, temp.compute_time);
    }
    delete[] data_list;
    return 0;
}