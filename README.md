# Acceleration of the Attention Mechanism on Ryzen AI using MLIR

**Submission for the Open Hardware Competition 2025** **Track:** Accelerated Computing: Unleash the potential of AMD ROCm supported Ryzen AI, and Radeon/Instinct GPUs  
**Author:** Angelos Bardouniotis (Supervisor: Asst. Prof. Christoforos Kachris)  
**Affiliation:** University of West Attica

---

### Abstract

This project provides a comprehensive performance analysis and acceleration of the Transformer attention mechanism on the AMD Ryzen AI Neural Processing Unit (NPU).
The primary objective was to evaluate the NPU's viability as a low-power, high-performance alternative to a CPU for on-device inference. Initial analysis revealed
that the available development toolchain precluded a pure NPU solution, leading to the key innovation of this work: a *hybrid CPU+NPU model*. In this model, the NPU
functions as a dedicated matrix multiplication accelerator controlled by C++ host logic. The results are definitive: the hybrid system achieved a peak end-to-end
speedup of over **129x** and an energy-to-solution improvement of up to **1000x** compared to the CPU baseline, validating the NPU's role in the modern on-device AI
landscape.

### Project Video

A complete 2-minute summary of our project is available on YouTube.

**[Link to final YouTube video here]**

---

### Key Results

Our hybrid CPU+NPU model demonstrates a significant leap in performance and energy efficiency over a traditional CPU-only baseline.

#### 129x End-to-End Speedup

The final end-to-end results for the full encoder pipeline demonstrate a peak performance gain of over **129x** for a sequence length of 512. This impressive scaling is a direct result of the workload becoming compute-bound, where the NPU’s massive computational advantage overcomes the fixed costs of data transfer.

![Speedup Graph](report/Full%20Encoder%20Speedup%20Factor%20(K_Model=768).jpg)
*Figure 1: Speedup factor of the hybrid NPU encoder pipeline compared to the CPU baseline.*

#### 1000x Energy-to-Solution Savings

For the BERT-level workload, the NPU was up to **1000x** more energy-efficient than the CPU. The total energy consumed (in Joules) was dramatically lower, confirming that the hybrid model is an exceptionally efficient solution for the entire Transformer architecture.

![Energy Graph](report/Total%20Energy%20Consumption%20to%20Solution%20NPU%20vs%20CPU.jpg)
*Figure 2: Total energy consumed by the NPU vs. the CPU.*
 up to **1000x** more energy-efficient than the CPU. The total energy consumed (in Joules) was dramatically lower, confirming that the hybrid model is an exceptionally efficient solution for the entire Transformer architecture.

---

### Introduction

The rapid evolution of artificial intelligence has been driven by increasingly complex neural network models, but their immense computational cost presents a significant barrier to deployment on consumer devices. This has spurred a revolution in computer architecture, leading to specialized hardware like the Neural Processing Unit (NPU), designed to execute AI workloads with far greater energy efficiency than traditional CPUs. This project addresses this critical intersection by conducting a comprehensive performance analysis and acceleration of the Transformer attention mechanism on a novel heterogeneous system, the AMD Ryzen AI NPU.

Initial investigation into the early-stage NPU development toolchain revealed significant challenges that made a "pure NPU" solution for the entire Transformer pipeline impractical. This challenge led to the key engineering innovation of this project: the development of a **hybrid CPU+NPU computational model**.

In this model, the NPU is leveraged as a highly specialized and efficient co-processor for its most optimal task (large matrix multiplications), while the general-purpose CPU handles the complex control flow and all other "glue logic". This pragmatic approach proved to be the only robust strategy for a successful implementation and became the core of our methodology.

### Methodology

Our methodology is centered on the **hybrid CPU+NPU model**, a pragmatic approach designed to maximize the strengths of each processor. The core strategy involves offloading only the most computationally intensive, matrix-multiplication-heavy components of the Transformer to the NPU, while the CPU retains control of all other "glue logic". This avoids the significant overhead associated with transferring data for smaller, less suitable workloads.

Two components clearly illustrate this design philosophy:

* **Parallel QKV Projections**: The initial Query, Key, and Value projections are three mathematically independent matrix multiplications. To maximize hardware utilization and hide the overhead of launching compute kernels, we executed these three operations in parallel on the NPU, with each launched from a separate C++ `std::thread` on the host.

* **Hybrid SDPA Block**: For the complex Scaled Dot-Product Attention block, the two large matrix multiplications were offloaded to the NPU, while the intermediate scaling and softmax operations remained on the CPU. This retained the flexibility of the CPU for non-matrix logic while leveraging the NPU for the heavy lifting.

This entire model was implemented using a two-path development workflow. A **Hardware Compilation Path** used Python and MLIR to generate a binary for the NPU, while a **Software Compilation Path** used C++ and CMake to build the host application that orchestrates the entire process.

### Results and Analysis

The performance gains of the hybrid model are substantial, reaching over **129x** the speed of the CPU baseline for the full encoder pipeline. This impressive scaling is a direct result of the workload becoming **compute-bound**; as the sequence length increases, the number of mathematical operations grows much faster than the amount of data that needs to be transferred, allowing the NPU's computational advantage to dominate the fixed cost of data transfer.

### Results and Analysis

The performance gains of the hybrid model are substantial, reaching over **129x** the speed of the CPU baseline for the full encoder pipeline. This impressive scaling is a direct result of the workload becoming **compute-bound**; as the sequence length increases, the number of mathematical operations grows much faster than the amount of data that needs to be transferred, allowing the NPU's computational advantage to dominate the fixed cost of data transfer.

A detailed breakdown of key components reveals where these gains originate:

| Component                 | Peak Speedup vs. CPU | Notes                                        |
| ------------------------- | -------------------- | -------------------------------------------- |
| **Parallel QKV Projections** | 1112.1x | Demonstrates benefit of host-side parallelism      |
| **Hybrid SDPA Block** | 14.4x                | Validates core hybrid CPU+NPU model          |
| **Hybrid FFN Block** | 214.96x | Highest component speedup; highly compute-bound |
| **Hybrid Cross-Attention** | 13.78x               | Proves hybrid model on complex decoder component   |
| **Full Encoder Pipeline** | **129.26x** | Final end-to-end system speedup              |
| **Full Decoder Pipeline** | **109.71x**              | Confirms performance on full decoder         |

The most critical result of this investigation is the analysis of energy efficiency. A fascinating insight from power measurements was that the NPU pipeline's instantaneous power draw (in Watts) was often **higher** than the CPU's, due to the high level of concurrent activity on the SoC.

However, because the NPU completed the task so much faster, the **total energy consumed** (in Joules) was dramatically lower. For the BERT-level workload, the NPU was up to **1000x more energy-efficient** than the CPU, a monumental advantage for any battery-powered or thermally constrained device.

### Conclusion

This project successfully demonstrated that a **hybrid CPU+NPU model** is a viable and highly effective strategy for accelerating Transformer workloads on modern, consumer-grade AI hardware. By achieving a speedup of over **129x** and an energy efficiency improvement of up to **1000x** compared to a standard CPU, the results show that for on-device applications where power, thermals, and physical constraints are the primary concerns, the AMD Ryzen AI NPU is a definitive champion.

The principal contribution of this work is a first-of-its-kind, in-depth performance and energy analysis of the Transformer architecture on this new class of hardware. It provides crucial, empirical data that validates the NPU's role in making powerful AI models practical for everyday devices.

---

### Hardware & Software Requirements

To replicate our results or run this project, you will need the following environment, which matches our specific experimental setup.

#### Hardware
* **CPU/NPU System**: AMD Ryzen 9 7940HS Processor.
* **System Memory**: 32 GB DDR5 RAM.
* **Operating System**: Ubuntu 22.04 LTS.

#### Software & Toolkits
* **Programming Languages**:
    * **C++23**: For the host application controlling the NPU.
    * **Python 3.10**: For the MLIR hardware generation scripts.
* **Key Libraries & Toolkits**:
    * **AMD Vitis Unified Software Platform 2023.2**: Provides the core `aie-compiler` and drivers for the NPU.
    * **Xilinx Runtime (XRT) 2.16**: The C++ library used by the host application to control the NPU hardware.
    * **CMake**: For building the C++ host application.

