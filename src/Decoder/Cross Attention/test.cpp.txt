#include "cxxopts.hpp"
#include <bits/stdc++.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <algorithm>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

// --- Data Type Definitions ---
#define DTYPE_IN int16_t
#define DTYPE_OUT int32_t
#define DTYPE_ACC int64_t

using IN_DATATYPE = DTYPE_IN;
using OUT_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;

// Helper function to initialize input vectors
template <typename T>
void initialize_vector(std::vector<T>& vec) {
    for (auto& val : vec) {
        val = static_cast<T>((rand() % 20) - 10);
    }
}

// Helper function to manually transpose a matrix
template <typename T>
void transpose_matrix(const std::vector<T>& input, std::vector<T>& output, int rows, int cols) {
    output.resize(cols * rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

int main(int argc, const char *argv[]) {
    // --- Command Line Argument Parsing ---
    cxxopts::Options options("Cross-Attention Test");
    options.add_options()
        ("xclbin1", "XCLBIN File for MatMul1 (QxK_T)", cxxopts::value<std::string>())
        ("insts1", "Instruction File for MatMul1", cxxopts::value<std::string>())
        ("xclbin2", "XCLBIN File for MatMul2 (Scores x V)", cxxopts::value<std::string>())
        ("insts2", "Instruction File for MatMul2", cxxopts::value<std::string>())
        ("k,kernel", "Kernel Name", cxxopts::value<std::string>()->default_value("MLIR_AIE"))
        ("v,verify", "Enable verification", cxxopts::value<bool>()->default_value("true"))
        ("iters", "Number of iterations", cxxopts::value<int>()->default_value("10"))
        ("warmup", "Number of warmup iterations", cxxopts::value<int>()->default_value("5"));
    
    auto vm = options.parse(argc, argv);

    int n_iterations = 10;
    int n_warmup_iterations = 5;

    // --- Cross-Attention Dimensions ---
    // MODIFICATION: M is now M_TGT (Target/Decoder Seq Len)
    const int M_TGT = M_DIM; 
    // MODIFICATION: Added M_SRC for the Source/Encoder Seq Len
    const int M_SRC = M_SRC_DIM; 
    const int K = K_DIM; // Head Dimension

    std::cout << "INFO: Running Cross-Attention Test." << std::endl;
    std::cout << "Dimensions: M_TGT=" << M_TGT << ", M_SRC=" << M_SRC << ", K=" << K << std::endl;

    srand(12345);

    // --- Vector Initialization ---
    // MODIFICATION: Updated vector sizes for cross-attention inputs
    std::vector<IN_DATATYPE> Q_Vec(M_TGT * K);      // Q comes from the target (decoder)
    std::vector<IN_DATATYPE> K_Vec(M_SRC * K);      // K comes from the source (encoder)
    std::vector<IN_DATATYPE> V_Vec(M_SRC * K);      // V comes from the source (encoder)
    
    initialize_vector(Q_Vec);
    initialize_vector(K_Vec);
    initialize_vector(V_Vec);

    // MODIFICATION: Transpose K from the source dimensions
    std::vector<IN_DATATYPE> K_T_Vec;
    transpose_matrix(K_Vec, K_T_Vec, M_SRC, K);

    // MODIFICATION: Attention scores matrix is now M_TGT x M_SRC
    std::vector<OUT_DATATYPE> MatMul1_Out(M_TGT * M_SRC);
    std::vector<IN_DATATYPE> Softmax_Out_16bit(M_TGT * M_SRC);
    std::vector<OUT_DATATYPE> Final_NPU_Out(M_TGT * K); // Final output has target seq len

    // --- XRT Setup ---
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);
    auto xclbin1 = xrt::xclbin(vm["xclbin1"].as<std::string>());
    std::vector<uint32_t> instr_v1 = test_utils::load_instr_binary(vm["insts1"].as<std::string>());
    device.register_xclbin(xclbin1);
    xrt::hw_context context1(device, xclbin1.get_uuid());
    auto kernel1 = xrt::kernel(context1, vm["kernel"].as<std::string>());
    auto xclbin2 = xrt::xclbin(vm["xclbin2"].as<std::string>());
    std::vector<uint32_t> instr_v2 = test_utils::load_instr_binary(vm["insts2"].as<std::string>());
    device.register_xclbin(xclbin2);
    xrt::hw_context context2(device, xclbin2.get_uuid());
    auto kernel2 = xrt::kernel(context2, vm["kernel"].as<std::string>());

    // --- Buffer Object Creation ---
    auto bo_instr1 = xrt::bo(device, instr_v1.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));
    auto bo_q = xrt::bo(device, Q_Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
    auto bo_k = xrt::bo(device, K_T_Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(4));
    auto bo_out1 = xrt::bo(device, MatMul1_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(5));
    auto bo_instr2 = xrt::bo(device, instr_v2.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel2.group_id(1));
    auto bo_sm_out = xrt::bo(device, Softmax_Out_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(3));
    auto bo_v = xrt::bo(device, V_Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(4));
    auto bo_out2 = xrt::bo(device, Final_NPU_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(5));

    bo_instr1.write(instr_v1.data());
    bo_instr2.write(instr_v2.data());
    bo_instr1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_instr2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // --- NPU PIPELINE TIMING ---
    float npu_time_total = 0;
    for (unsigned iter = 0; iter < n_iterations + n_warmup_iterations; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        // --- MatMul1 (Q @ K_T) on NPU ---
        bo_q.write(Q_Vec.data());
        bo_k.write(K_T_Vec.data());
        bo_q.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_k.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run1 = kernel1(3, bo_instr1, instr_v1.size(), bo_q, bo_k, bo_out1);
        run1.wait();
        bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out1.read(MatMul1_Out.data());

        // --- Scaling and Softmax on CPU ---
        // MODIFICATION: No causal mask is used in cross-attention.
        float scale = 1.0f / sqrt(K);
        for(int i = 0; i < M_TGT; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M_SRC; ++j) max_val = std::max(max_val, static_cast<float>(MatMul1_Out[i * M_SRC + j]));
            float sum_exp = 0.0f;
            for(int j = 0; j < M_SRC; ++j) sum_exp += expf((static_cast<float>(MatMul1_Out[i * M_SRC + j]) * scale) - max_val);
            for(int j = 0; j < M_SRC; ++j) {
                float softmax_val = expf((static_cast<float>(MatMul1_Out[i * M_SRC + j]) * scale) - max_val) / sum_exp;
                Softmax_Out_16bit[i * M_SRC + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
            }
        }

        // --- MatMul2 (Scores @ V) on NPU ---
        bo_sm_out.write(Softmax_Out_16bit.data());
        bo_v.write(V_Vec.data());
        bo_sm_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run2 = kernel2(3, bo_instr2, instr_v2.size(), bo_sm_out, bo_v, bo_out2);
        run2.wait();
        bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out2.read(Final_NPU_Out.data());

        if (iter >= n_warmup_iterations) {
            npu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
        }
    }

    // --- PURE CPU TIMING BLOCK ---
    std::cout << "\n--- Running CPU reference for timing comparison ---" << std::endl;
    float cpu_time_total = 0;
    std::vector<OUT_DATATYPE> Final_CPU_Out(M_TGT * K);
    for (unsigned iter = 0; iter < n_iterations + n_warmup_iterations; ++iter) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        // MODIFICATION: CPU reference loops updated for new dimensions
        std::vector<OUT_DATATYPE> cpu_matmul1_out(M_TGT * M_SRC);
        for (int i = 0; i < M_TGT; ++i) {
            for (int j = 0; j < M_SRC; ++j) {
                ACC_DATATYPE acc = 0;
                for (int k = 0; k < K; ++k) {
                    acc += (ACC_DATATYPE)Q_Vec[i * K + k] * (ACC_DATATYPE)K_Vec[j * K + k];
                }
                cpu_matmul1_out[i * M_SRC + j] = (OUT_DATATYPE)acc;
            }
        }

        std::vector<IN_DATATYPE> cpu_softmax_out_16bit(M_TGT * M_SRC);
        float scale = 1.0f / sqrt(K);
        for(int i = 0; i < M_TGT; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M_SRC; ++j) max_val = std::max(max_val, static_cast<float>(cpu_matmul1_out[i * M_SRC + j]));
            float sum_exp = 0.0f;
            for(int j = 0; j < M_SRC; ++j) sum_exp += expf((static_cast<float>(cpu_matmul1_out[i * M_SRC + j]) * scale) - max_val);
            for(int j = 0; j < M_SRC; ++j) {
                float softmax_val = expf((static_cast<float>(cpu_matmul1_out[i * M_SRC + j]) * scale) - max_val) / sum_exp;
                cpu_softmax_out_16bit[i * M_SRC + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
            }
        }
        
        for(int i = 0; i < M_TGT; ++i) {
            for(int j = 0; j < K; ++j) {
                ACC_DATATYPE acc = 0;
                for(int k = 0; k < M_SRC; ++k) {
                    acc += (ACC_DATATYPE)cpu_softmax_out_16bit[i * M_SRC + k] * (ACC_DATATYPE)V_Vec[k * K + j];
                }
                Final_CPU_Out[i * K + j] = (OUT_DATATYPE)acc;
            }
        }
        
        if (iter >= n_warmup_iterations) {
            cpu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - cpu_start).count();
        }
    }

    // --- FINAL RESULTS AND VERIFICATION ---
    std::cout << "\n--- Final Performance ---" << std::endl;
    std::cout << "Avg NPU Pipeline time: " << (npu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    std::cout << "Avg CPU Pipeline time: " << (cpu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    int errors = 0;
    if (vm["verify"].as<bool>()) {
        for(int i=0; i< M_TGT * K; ++i) {
            if(abs(Final_NPU_Out[i] - Final_CPU_Out[i]) > 2) { 
                if(errors < 10) { 
                    std::cout << "ERROR: index=" << i << ", npu=" << Final_NPU_Out[i]
                              << ", ref=" << Final_CPU_Out[i] << std::endl;
                }
                errors++;
            }
        }
    }

    if (!errors) {
        std::cout << "\nPASS! (NPU pipeline result matches CPU reference)\n\n";
        return 0;
    } else {
        std::cout << "\nFailed. Error count: " << errors << "\n\n";
        return 1;
    }
}
