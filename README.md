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

![Speedup Graph](report/speedup_graph.png)
*Figure 1: Speedup factor of the hybrid NPU encoder pipeline compared to the CPU baseline.*

#### 1000x Energy-to-Solution Savings

For the BERT-level workload, the NPU was up to **1000x** more energy-efficient than the CPU. The total energy consumed (in Joules) was dramatically lower, confirming that the hybrid model is an exceptionally efficient solution for the entire Transformer architecture.

![Energy Graph](report/energy_graph.png)
*Figure 2: Total energy consumed by the NPU vs. the CPU.*
