# Coalescing Penalty: The Cost of Chaos

## Benchmark Results (100M Integers)

| Pattern | Time (ms) | Speedup vs Baseline | Notes |
| :--- | :--- | :--- | :--- |
| **Contiguous (k=1)** | 4.20 | 1.0x | The Gold Standard (100% Bandwidth) |
| **Strided (k=2)** | 6.51 | 0.64x | 50% Cache Line Efficiency |
| **Strided (k=32)** | 20.23 | 0.21x | Worst case for Cache Lines (1 transaction per thread) |
| **TLB Wall (k=128)** | 49.83 | 0.08x | Missing the Cache + Missing the Page (TLB) |
| **Random (Gather)** | 102.99 | 0.04x | Complete TLB Thrashing + No Prefetching |

## Key Takeaway
Memory access pattern matters more than compute power. 
Random access is **25x slower** than sequential access on this hardware.