#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void prefix_sum(int *a, int *d_o, int N, int prev_sum) {
  __shared__ int tile[1024];
  int block_offset_x = blockIdx.x * blockDim.x;
  int tx = (int)threadIdx.x;
  int index = block_offset_x + tx;
  if (index < N)
    tile[tx] = a[index];
  if (tx == 0)
    tile[tx] += prev_sum;
  __syncthreads();
  for (int step = 1; step <= 512; step *= 2) {
    int val = 0;
    if (index < N)
      val = tile[tx];
    if (tx - step >= 0 && tx - step < N) {
      val += tile[tx - step];
    }
    __syncthreads();
    if (index < N) {
      tile[tx] = val;
    }
    __syncthreads();
  }
  if (index < N)
    d_o[index] = tile[tx];
}

__global__ void prefix_sum_optimal(int *a, int *d_o, int N) {
  __shared__ int tile[1024];
  int bx = blockIdx.x;
  int block_offset = bx * blockDim.x;
  int index = block_offset + threadIdx.x;
  int tx = threadIdx.x;
  if (index < N)
    tile[tx] = a[index];
  __syncthreads();

  for (int i = 1; i <= 512; i *= 2) {
    int temp = 0;
    if (index < N && tx - i >= 0) {
      temp += tile[tx - i];
    }
    __syncthreads();
    if (index < N) {
      tile[tx] += temp;
    }
    __syncthreads();
  }

  if (index < N && tx == N - 1)
    d_o[bx] = tile[tx];
  if (index < N)
    a[index] = tile[tx];
}

void print(int *a, int N) {
  printf("\n");
  for (int i = 0; i < N; i += 1) {
    printf("%d ", a[i]);
  }
  printf("\n");
}

__global__ void add_it(int *a, int *d_o, int N) {
  int bx = blockIdx.x;
  int block_offset = bx * blockDim.x;
  int tx = threadIdx.x;
  int index = block_offset + tx;
  int val = 0;
  if (bx == 0)
    val = d_o[bx];
  // printf("bx:- %d, val:- %d\n", bx, val);
  if (index < N)
    a[index] += val;
}

int *test_prefix_optimal(int N = 100) {

  int *a, *d_a;
  int bytes = N * sizeof(int);
  cudaMallocHost((void **)&a, bytes);
  cudaMalloc((void **)&d_a, bytes);

  for (int i = 0; i < N; i += 1)
    a[i] = (i + 1);
  // printf("Original:-\n");
  // print(a, N);

  int max_threads = 1024;
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  int blocks = (N + max_threads - 1) / max_threads;
  int *output, *d_o;
  cudaMallocHost((void **)&output, blocks * sizeof(int));
  cudaMalloc((void **)&d_o, blocks * sizeof(int));
  cudaEventRecord(start, 0);
  cudaStream_t streams[blocks];
  for (int i = 0; i < blocks; i += 1) {
    cudaStreamCreate(&streams[i]);
  }
  for (int i = 0; i < blocks; i += 1) {
    int steps = max_threads;
    if (N < max_threads)
      steps = N;
    else if (N - (max_threads * i) < max_threads)
      steps = N - (max_threads * i);
    int prev_steps = i * max_threads;
    int block_bytes = steps * sizeof(int);
    cudaMemcpyAsync(d_a + prev_steps, a + prev_steps, block_bytes,
                    cudaMemcpyHostToDevice, streams[i]);
    prefix_sum_optimal<<<1, max_threads, 0, streams[i]>>>(d_a + prev_steps,
                                                          d_o + i, steps);
    cudaMemcpyAsync(output + i, d_o + i, sizeof(int), cudaMemcpyDeviceToHost,
                    streams[i]);
  }
  cudaDeviceSynchronize();
  int temp = 0;
  for (int i = 0; i < blocks; i += 1) {
    int temp1 = output[i];
    output[i] = temp;
    temp += temp1;
  }
  output[0] = 0;
  for (int i = 0; i < blocks; i += 1) {
    int steps = max_threads;
    if (N < max_threads)
      steps = N;
    else if (N - (max_threads * i) < max_threads)
      steps = N - (max_threads * i);
    int prev_steps = i * max_threads;
    int block_bytes = steps * sizeof(int);
    cudaMemcpyAsync(d_o + i, output + i, sizeof(int), cudaMemcpyHostToDevice,
                    streams[i]);
    add_it<<<1, max_threads, 0, streams[i]>>>(d_a + prev_steps, d_o + i, steps);
    cudaMemcpyAsync(a + prev_steps, d_a + prev_steps, block_bytes,
                    cudaMemcpyDeviceToHost, streams[i]);
  }
  cudaDeviceSynchronize();
  // print(a, N);
  int actaul_val = (N * (N + 1)) / 2;
  printf("last index:- %d\nActual:- %d\n", a[N - 1], actaul_val);
  return 0;
  cudaFreeHost(a);
  cudaFree(d_a);
  return a;
}

void test_prefix_sum(int N = 100) {
  int *a, *b, *d_a, *d_b, *output, *d_o;
  int total = N;
  int bytes = total * sizeof(int);
  cudaMallocHost((void **)&a, bytes);
  cudaMallocHost((void **)&output, bytes);
  cudaMalloc((void **)&d_a, bytes);
  cudaMalloc((void **)&d_o, bytes);
  cudaMemset(d_o, 0, bytes);
  for (int i = 0; i < total; i += 1) {
    a[i] = i + 1;
  }
  int max_threads = 1024;
  int blocks = (N + max_threads - 1) / max_threads;
  int prev_sum = 0;
  for (int i = 0; i < blocks; i += 1) {
    int threads_offset = (i * 1024);
    int rem_element = N - threads_offset;
    int chunk = std::min(rem_element, max_threads);
    int last_element = chunk - 1;
    cudaMemcpy(d_a + threads_offset, a + threads_offset, sizeof(int) * chunk,
               cudaMemcpyHostToDevice);
    prefix_sum<<<1, max_threads>>>(d_a + threads_offset, d_o + threads_offset,
                                   rem_element, prev_sum);
    cudaDeviceSynchronize();
    cudaMemcpy(output + threads_offset, d_o + threads_offset,
               sizeof(int) * chunk, cudaMemcpyDeviceToHost);
    prev_sum = output[threads_offset + last_element];
  }
  int actaul_val = (N * (N + 1)) / 2;
  printf("last index:- %d\nActual:- %d\n", output[N - 1], actaul_val);
}

int main() {
  test_prefix_sum(27293);
  test_prefix_optimal(3000)
}