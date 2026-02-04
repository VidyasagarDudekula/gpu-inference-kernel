# The transfer speed from Host to Device, Pinned vs Pagged memory allocation in Host

### The gpu hates to have some refereces to the memory address, it always deals with direct memory.
### So the way you allocate the Memory in host will decide the transfer speed of that data from Host to Device.
### When you allocate a memory using new/malloc --> this is a pagged memory allocation, as CPU can handle refereced memory via page tables.
### The same bad for GPU while cudaMemcpy as it will ask the os to stop everything and move that referenced data (pagged) to buffer (pinned) from there it will move the data to gpu.
### While cudaMallocHost is a pinned memory allocation, so it a direct un modified transfer to the gpu from host.


## Check out few stats based on array size and pinned/pagged
N:- 100, isPinned:- 1, elapsed_time:- 0.015232
N:- 100, isPinned:- 0, elapsed_time:- 0.009216
N:- 1024, isPinned:- 1, elapsed_time:- 0.009856
N:- 1024, isPinned:- 0, elapsed_time:- 0.005728
N:- 10240, isPinned:- 1, elapsed_time:- 0.011296
N:- 10240, isPinned:- 0, elapsed_time:- 0.020512
N:- 102400, isPinned:- 1, elapsed_time:- 0.039232
N:- 102400, isPinned:- 0, elapsed_time:- 0.148352
N:- 1024000, isPinned:- 1, elapsed_time:- 0.348096
N:- 1024000, isPinned:- 0, elapsed_time:- 1.320960
N:- 10240000, isPinned:- 1, elapsed_time:- 3.198560
N:- 10240000, isPinned:- 0, elapsed_time:- 10.941856