#include<iostream>
#include<cuda.h>

int main(){
    FILE *filePointer;
    filePointer = fopen("gpu-inference-kernel/docs/hardware.md", "w");

    if (filePointer == NULL) {
        perror("Error opening file");
        return 1;
    }


    int devices;
    cudaGetDeviceCount(&devices);
    fprintf(filePointer, "# Current GPU properties\n");
    for(int i=0; i<devices; i+=1){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        fprintf(filePointer, "## Device:- %d\n", i);
        fprintf(filePointer, "- Device name:- %s\n", prop.name);
        fprintf(filePointer, "- Number of SMs:- %d\n", prop.multiProcessorCount);
        fprintf(filePointer, "- Max threads for block:- %d\n", prop.maxThreadsPerBlock);
        fprintf(filePointer, "- Total Global memory:- %zu\n", prop.totalGlobalMem);
        fprintf(filePointer, "- Total Shared memory:- %zu\n", prop.sharedMemPerBlock);
        fprintf(filePointer, "- Warp Size:- %d\n", prop.warpSize);
        fprintf(filePointer, "\n**************\n");
    }
    fclose(filePointer);
}