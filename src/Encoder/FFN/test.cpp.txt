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

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h" // Includes matmul_common::verify

// --- Data Type Definitions ---
#define DTYPE_IN int16_t
#define DTYPE_OUT int32_t
#define DTYPE_ACC int64_t

using IN_DATATYPE = DTYPE_IN;
using OUT_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;

// Helper function to initialize input vectors with small, deterministic random numbers
template <typename T>
void initialize_vector(std::vector<T>& vec) {
    for (auto& val : vec) {
        val = static_cast<T>((rand() % 20) - 10);
    }
}

int main(int argc, const char *argv[]) {
    cxxopts::Options options("FFN Test");
    options.add_options()
        ("xclbin1", "XCLBIN File for MatMul1", cxxopts::value<std::string>())
        ("insts1", "Instruction File for MatMul1", cxxopts::value<std::string>())
        ("xclbin2", "XCLBIN File for MatMul2", cxxopts::value<std::string>())
        ("insts2", "Instruction File for MatMul2", cxxopts::value<std::string>())
        ("k,kernel", "Kernel Name", cxxopts::value<std::string>()->default_value("MLIR_AIE"))
        ("v,verify", "Enable verification", cxxopts::value<bool>()->default_value("true"))
        ("iters", "Number of iterations", cxxopts::value<int>()->default_value("10"))
        ("warmup", "Number of warmup iterations", cxxopts::value<int>()->default_value("5"))
        ("verbosity", "Verbosity Level", cxxopts::value<int>()->default_value("0"));
    
    auto vm = options.parse(argc, argv);

    int verbosity = vm["verbosity"].as<int>();
    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();

    // --- FFN Dimensions (passed from Makefile via compiler definitions) ---
    const int M = M_DIM;
    const int K1 = K1_DIM;
    const int N1 = N1_DIM;
    const int K2 = K2_DIM;
    const int N2 = N2_DIM;

    // Use a fixed seed for deterministic results
    srand(12345);

    // --- Vector Initialization ---
    std::vector<IN_DATATYPE> InVec(M * K1);
    std::vector<IN_DATATYPE> W1Vec(K1 * N1), B1Vec(N1);
    std::vector<IN_DATATYPE> W2Vec(K2 * N2), B2Vec(N2);
    initialize_vector(InVec);
    initialize_vector(W1Vec);
    initialize_vector(B1Vec);
    initialize_vector(W2Vec);
    initialize_vector(B2Vec);

    std::vector<OUT_DATATYPE> MatMul1_Out(M * N1);
    std::vector<IN_DATATYPE> MatMul2_In_16bit(M * K2);
    std::vector<OUT_DATATYPE> Final_NPU_Out(M * N2);

    // --- XRT Setup (Done ONCE before the loop) ---
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

    // --- Buffer Object Creation (Done ONCE before the loop) ---
    auto bo_instr1 = xrt::bo(device, instr_v1.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));
    auto bo_in1 = xrt::bo(device, InVec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
    auto bo_w1 = xrt::bo(device, W1Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(4));
    auto bo_out1 = xrt::bo(device, MatMul1_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(5));

    auto bo_instr2 = xrt::bo(device, instr_v2.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel2.group_id(1));
    auto bo_in2 = xrt::bo(device, MatMul2_In_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(3));
    auto bo_w2 = xrt::bo(device, W2Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(4));
    auto bo_out2 = xrt::bo(device, Final_NPU_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(5));

    bo_instr1.write(instr_v1.data());
    bo_instr2.write(instr_v2.data());
    bo_instr1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_instr2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // --- NPU PIPELINE TIMING ---
    float npu_time_total = 0;
    unsigned num_iter = n_iterations + n_warmup_iterations;

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();

        bo_in1.write(InVec.data());
        bo_w1.write(W1Vec.data());
        bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run1 = kernel1(3, bo_instr1, instr_v1.size(), bo_in1, bo_w1, bo_out1);
        run1.wait();
        bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out1.read(MatMul1_Out.data());

        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < N1; ++j) {
                int index = i * N1 + j;
                ACC_DATATYPE biased_val_64 = (ACC_DATATYPE)MatMul1_Out[index] + B1Vec[j];
                OUT_DATATYPE relu_32bit = std::max((OUT_DATATYPE)0, (OUT_DATATYPE)biased_val_64);
                MatMul2_In_16bit[index] = static_cast<IN_DATATYPE>(std::clamp(relu_32bit, (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
            }
        }

        bo_in2.write(MatMul2_In_16bit.data());
        bo_w2.write(W2Vec.data());
        bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run2 = kernel2(3, bo_instr2, instr_v2.size(), bo_in2, bo_w2, bo_out2);
        run2.wait();
        bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out2.read(Final_NPU_Out.data());

        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < N2; ++j) {
                Final_NPU_Out[i * N2 + j] += B2Vec[j];
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        if (iter >= n_warmup_iterations) {
            npu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        }
    }

    // --- PURE CPU TIMING BLOCK ---
    std::cout << "\n--- Running CPU reference for timing comparison ---" << std::endl;
    float cpu_time_total = 0;
    std::vector<OUT_DATATYPE> Final_CPU_Out(M * N2);

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        std::vector<OUT_DATATYPE> cpu_matmul1_out(M * N1);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N1; ++j) {
                ACC_DATATYPE acc = 0;
                for (int k = 0; k < K1; ++k) {
                    acc += (ACC_DATATYPE)InVec[i * K1 + k] * (ACC_DATATYPE)W1Vec[k * N1 + j];
                }
                cpu_matmul1_out[i * N1 + j] = (OUT_DATATYPE)acc;
            }
        }

        std::vector<IN_DATATYPE> cpu_relu_out_16bit(M * N1);
        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < N1; ++j) {
                int index = i * N1 + j;
                ACC_DATATYPE acc = (ACC_DATATYPE)cpu_matmul1_out[index] + B1Vec[j];
                OUT_DATATYPE relu_32bit = std::max((OUT_DATATYPE)0, (OUT_DATATYPE)acc);
                cpu_relu_out_16bit[index] = static_cast<IN_DATATYPE>(std::clamp(relu_32bit, (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
            }
        }
        
        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < N2; ++j) {
                ACC_DATATYPE acc = 0;
                for(int k = 0; k < K2; ++k) {
                    acc += (ACC_DATATYPE)cpu_relu_out_16bit[i * K2 + k] * (ACC_DATATYPE)W2Vec[k * N2 + j];
                }
                Final_CPU_Out[i * N2 + j] = (OUT_DATATYPE)acc + B2Vec[j];
            }
        }
        
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        if (iter >= n_warmup_iterations) {
            cpu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
        }
    }

    // --- FINAL RESULTS ---
    std::cout << "\n--- Final Performance ---" << std::endl;
    std::cout << "Avg NPU Pipeline time: " << (npu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    std::cout << "Avg CPU Pipeline time: " << (cpu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;

    // --- FINAL VERIFICATION ---
    int errors = 0;
    if(do_verify) {
        for(int i=0; i< M * N2; ++i) {
            if(Final_NPU_Out[i] != Final_CPU_Out[i]) {
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
