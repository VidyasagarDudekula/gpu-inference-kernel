# Timeline Analysis: Vector Add (10M Elements)

## 1. For 100M data
* **Copy H2D:** 31.64 ms
* **Kernel:** 2.11 ms
* **Copy D2H:** 31.68 ms
* **Total:** 65.45 ms

## Percentage:-
* **Compute Percentage:** 3.22%
* **Transfer Percentage:** 96.78%

## 2. The Timeline Visualization
(Time flows from Left -> Right)

[CPU]   | <Driver Overhead> |
[Copy]  | [================ H2D Copy =================]                 [================ D2H Copy =================]
[GPU]   |                                              [== Kernel ==]

## 3. Analysis
* **Bottleneck:** The PCIe Bus (Bandwidth). The transfer takes ~30x longer than the computation. Looking at the analysis its a clear memory bound, most of the time GPU is just waiting for the memory to arrive.
* **Observation:** Well, one thing at the time of copy from host -> device, the CPU is idle, the time of data from device -> host GPU os idle, maybe we can optimize this.

## Timing Table (Vector Add)

> All times are in **milliseconds (ms)**.

| N (elements)   | Pinned Memory | Total (ms)  | H2D (ms)   | D2H (ms)   | Kernel (ms) |
|---------------:|:-------------:|------------:|-----------:|-----------:|------------:|
|           100  |      Yes      |   0.122464  |  0.015360  |  0.006816  |   0.095232  |
|           100  |      No       |   0.025056  |  0.008352  |  0.006336  |   0.005152  |
|         1,024  |      Yes      |   0.030720  |  0.009024  |  0.011040  |   0.005856  |
|         1,024  |      No       |   0.021344  |  0.006208  |  0.006272  |   0.004096  |
|        10,240  |      Yes      |   0.034560  |  0.010496  |  0.012928  |   0.006400  |
|        10,240  |      No       |   0.063648  |  0.020192  |  0.035584  |   0.003072  |
|       102,400  |      Yes      |   0.091648  |  0.038624  |  0.041792  |   0.006464  |
|       102,400  |      No       |   0.420736  |  0.151232  |  0.254272  |   0.010016  |
|     1,024,000  |      Yes      |   0.670528  |  0.319136  |  0.321568  |   0.024672  |
|     1,024,000  |      No       |   3.493696  |  1.132896  |  2.331040  |   0.022880  |
|    10,240,000  |      Yes      |   6.497792  |  3.150112  |  3.123328  |   0.218816  |
|    10,240,000  |      No       |  32.936481  | 10.276864  | 22.440577  |   0.212320  |
|   102,400,000  |      Yes      |  65.453346  | 31.647167  | 31.685440  |   2.114432  |
|   102,400,000  |      No       | 323.581177  |101.635391  |219.842880  |   2.097120  |
