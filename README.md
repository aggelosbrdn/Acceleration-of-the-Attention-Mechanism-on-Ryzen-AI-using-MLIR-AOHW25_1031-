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
