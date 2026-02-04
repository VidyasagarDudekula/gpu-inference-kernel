# Host-to-Device Transfer Speed: Pinned vs Paged Host Memory

## Why allocation type affects transfer speed

- GPUs prefer working with **direct (non-pageable) memory**.
- The way you allocate memory on the host affects how fast data can be transferred from **Host → Device**.

### Paged (pageable) host memory: `new` / `malloc`

- Allocating with `new` or `malloc` gives **paged/pageable** memory.
- The CPU can handle pageable memory via **page tables**.
- For a `cudaMemcpy` from pageable host memory:
  - CUDA typically has to **stage the data** through a temporary **pinned buffer**.
  - This adds overhead before the data can be copied to the GPU.

### Pinned (page-locked) host memory: `cudaMallocHost`

- Allocating with `cudaMallocHost` gives **pinned (page-locked)** memory.
- Since it’s already pinned, CUDA can transfer it more directly to the GPU (often faster, especially for larger transfers).

---

## Stats: array size vs pinned/paged

| N (elements) | isPinned | elapsed_time |
|---:|---:|---:|
| 100 | 1 | 0.015232 |
| 100 | 0 | 0.009216 |
| 1024 | 1 | 0.009856 |
| 1024 | 0 | 0.005728 |
| 10240 | 1 | 0.011296 |
| 10240 | 0 | 0.020512 |
| 102400 | 1 | 0.039232 |
| 102400 | 0 | 0.148352 |
| 1024000 | 1 | 0.348096 |
| 1024000 | 0 | 1.320960 |
| 10240000 | 1 | 3.198560 |
| 10240000 | 0 | 10.941856 |
