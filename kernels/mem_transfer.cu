#include<iostream>
#include<cuda.h>

struct customeDict{
    int n_value;
    bool is_pinned;
    float time_took;
};

int main(){
    int *a, *d_a;
    int N_list[] = {100, 1024, 1024*10, 1024*100, 1024*1000, 1024*10000};
    bool pinned_list[] = {true, false};
    int total_values = int(std::size(N_list)) * int(std::size(pinned_list));
    customeDict *data_list = new customeDict[total_values];
    int counter = 0;
    for(auto N: N_list){
        int bytes = N  * sizeof(int);
        for(auto isPinned: pinned_list){
            cudaEvent_t start, end;
            float elapsed_time;

            if(isPinned)
                cudaMallocHost((void **)&a, bytes);
            else
                a = new int[N];
            cudaMalloc((void **)&d_a, bytes);

            cudaEventCreate(&start);
            cudaEventCreate(&end);
            cudaEventRecord(start, 0);

            cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(end, 0);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed_time, start, end);
            customeDict temp;
            temp.n_value = N;
            temp.is_pinned = isPinned;
            temp.time_took = elapsed_time;
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
        printf("N:- %d, isPinned:- %d, elapsed_time:- %f\n", temp.n_value, temp.is_pinned, temp.time_took);
    }
    delete[] data_list;
    return 0;
}