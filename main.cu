// #include<iostream>
// #include <nvbench/nvbench.cuh>

// __global__ void my_kernel(){
//     int arr[] = {1, 2, 3, 4, 5};
//     size_t size_of_array = sizeof(arr)/sizeof(arr[0]);
//     printf("Size of array %d\n", size_of_array);
// }

// void my_benchmark(nvbench::state &state){
//     state.exec(nvbench::exec_tag::sync, [](nvbench::launch& launch){
//         my_kernel<<<1, 2>>>();
//     });
// }

// NVBENCH_BENCH(my_benchmark);

// NVBENCH_MAIN