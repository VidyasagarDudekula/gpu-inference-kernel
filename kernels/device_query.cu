#include<iostream>
#include<cuda.h>

int main(){
    int devices;
    cudaGetDeviceCount(&devices);
    printf("# Current GPU properties\n");
    for(int i=0; i<devices; i+=1){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("## Device:- %d\n", i);
        printf("- Device name:- %s\n", prop.name);
        printf("- Number of SMs:- %d\n", prop.multiProcessorCount);
        printf("- Max threads for block:- %d\n", prop.maxThreadsPerBlock);
        printf("- Total Global memory:- %zu\n", prop.totalGlobalMem);
        printf("- Total Shared memory:- %zu\n", prop.sharedMemPerBlock);
        printf("- Warp Size:- %d\n", prop.warpSize);
        printf("\n**************\n");
    }
}