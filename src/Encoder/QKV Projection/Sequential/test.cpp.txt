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
#include <cstdio> // Required for popen

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

// Helper function for CPU-based matrix multiplication
void cpu_matmul(const std::vector<IN_DATATYPE>& A, const std::vector<IN_DATATYPE>& B, std::vector<OUT_DATATYPE>& C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            ACC_DATATYPE acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += (ACC_DATATYPE)A[i * K + k] * (ACC_DATATYPE)B[k * N + j];
            }
            C[i * N + j] = (OUT_DATATYPE)acc;
        }
    }
}

// Helper function to get power consumption
float get_power_in_watts() {
    std::string command = "sensors | grep PPT";
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return -1.0; // Return error
    }
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }
    pclose(pipe);

    std::stringstream ss(result);
    std::string temp;
    float power = -1.0;
    while (ss >> temp) {
        if (std::stringstream(temp) >> power) {
            return power;
        }
    }
    return -1.0; // Return error if not found
}


int main(int argc, const char *argv[]) {
    // --- Command Line Argument Parsing ---
    cxxopts::Options options("Sequential MatMul Test");
    options.add_options()
        ("xclbin1", "XCLBIN File for MatMul1", cxxopts::value<std::string>())
        ("insts1", "Instruction File for MatMul1", cxxopts::value<std::string>())
        ("xclbin2", "XCLBIN File for MatMul2", cxxopts::value<std::string>())
        ("insts2", "Instruction File for MatMul2", cxxopts::value<std::string>())
        ("xclbin3", "XCLBIN File for MatMul3", cxxopts::value<std::string>())
        ("insts3", "Instruction File for MatMul3", cxxopts::value<std::string>())
        ("k,kernel", "Kernel Name", cxxopts::value<std::string>()->default_value("MLIR_AIE"))
        ("v,verify", "Enable verification", cxxopts::value<bool>()->default_value("true"))
        ("iters", "Number of iterations", cxxopts::value<int>()->default_value("10"))
        ("warmup", "Number of warmup iterations", cxxopts::value<int>()->default_value("5"))
        ("verbosity", "Verbosity Level", cxxopts::value<int>()->default_value("0"));
    
    auto vm = options.parse(argc, argv);

    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();

    const int M = M_DIM;
    const int K1 = K1_DIM;
    const int N1 = N1_DIM;
    const int N2 = N2_DIM;

    srand(12345);

    std::vector<IN_DATATYPE> InVec(M * K1);
    std::vector<IN_DATATYPE> W1Vec(K1 * N1);
    std::vector<IN_DATATYPE> W2Vec(N1 * N2);
    std::vector<IN_DATATYPE> W3Vec(N2 * K1);
    
    initialize_vector(InVec);
    initialize_vector(W1Vec);
    initialize_vector(W2Vec);
    initialize_vector(W3Vec);

    std::vector<OUT_DATATYPE> MatMul1_Out(M * N1);
    std::vector<IN_DATATYPE> MatMul2_In(M * N1);
    std::vector<OUT_DATATYPE> MatMul2_Out(M * N2);
    std::vector<IN_DATATYPE> MatMul3_In(M * N2);
    std::vector<OUT_DATATYPE> Final_NPU_Out(M * K1);

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

    auto xclbin3 = xrt::xclbin(vm["xclbin3"].as<std::string>());
    std::vector<uint32_t> instr_v3 = test_utils::load_instr_binary(vm["insts3"].as<std::string>());
    device.register_xclbin(xclbin3);
    xrt::hw_context context3(device, xclbin3.get_uuid());
    auto kernel3 = xrt::kernel(context3, vm["kernel"].as<std::string>());

    auto bo_instr1 = xrt::bo(device, instr_v1.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));
    auto bo_in1 = xrt::bo(device, InVec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
    auto bo_w1 = xrt::bo(device, W1Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(4));
    auto bo_out1 = xrt::bo(device, MatMul1_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(5));

    auto bo_instr2 = xrt::bo(device, instr_v2.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel2.group_id(1));
    auto bo_in2 = xrt::bo(device, MatMul2_In.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(3));
    auto bo_w2 = xrt::bo(device, W2Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(4));
    auto bo_out2 = xrt::bo(device, MatMul2_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel2.group_id(5));

    auto bo_instr3 = xrt::bo(device, instr_v3.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel3.group_id(1));
    auto bo_in3 = xrt::bo(device, MatMul3_In.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel3.group_id(3));
    auto bo_w3 = xrt::bo(device, W3Vec.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel3.group_id(4));
    auto bo_out3 = xrt::bo(device, Final_NPU_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel3.group_id(5));

    bo_instr1.write(instr_v1.data());
    bo_instr2.write(instr_v2.data());
    bo_instr3.write(instr_v3.data());
    bo_instr1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_instr2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_instr3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // --- NPU PIPELINE TIMING ---
    float npu_time_total = 0;
    float npu_power_total = 0;
    unsigned num_iter = n_iterations + n_warmup_iterations;

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        float power_start = get_power_in_watts();
        auto start = std::chrono::high_resolution_clock::now();

        bo_in1.write(InVec.data()); bo_w1.write(W1Vec.data());
        bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE); bo_w1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run1 = kernel1(3, bo_instr1, instr_v1.size(), bo_in1, bo_w1, bo_out1);
        run1.wait();
        bo_out1.sync(XCL_BO_SYNC_BO_FROM_DEVICE); bo_out1.read(MatMul1_Out.data());
        for(int i=0; i < M * N1; ++i) MatMul2_In[i] = static_cast<IN_DATATYPE>(std::clamp(MatMul1_Out[i], (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
        bo_in2.write(MatMul2_In.data()); bo_w2.write(W2Vec.data());
        bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE); bo_w2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run2 = kernel2(3, bo_instr2, instr_v2.size(), bo_in2, bo_w2, bo_out2);
        run2.wait();
        bo_out2.sync(XCL_BO_SYNC_BO_FROM_DEVICE); bo_out2.read(MatMul2_Out.data());
        for(int i=0; i < M * N2; ++i) MatMul3_In[i] = static_cast<IN_DATATYPE>(std::clamp(MatMul2_Out[i], (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
        bo_in3.write(MatMul3_In.data()); bo_w3.write(W3Vec.data());
        bo_in3.sync(XCL_BO_SYNC_BO_TO_DEVICE); bo_w3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run3 = kernel3(3, bo_instr3, instr_v3.size(), bo_in3, bo_w3, bo_out3);
        run3.wait();
        bo_out3.sync(XCL_BO_SYNC_BO_FROM_DEVICE); bo_out3.read(Final_NPU_Out.data());

        auto stop = std::chrono::high_resolution_clock::now();
        float power_stop = get_power_in_watts();
        
        if (iter >= n_warmup_iterations) {
            npu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            npu_power_total += (power_start + power_stop) / 2.0;
        }
    }

    // --- PURE CPU TIMING BLOCK ---
    std::cout << "\n--- Running CPU reference for timing comparison ---" << std::endl;
    float cpu_time_total = 0;
    float cpu_power_total = 0;
    std::vector<OUT_DATATYPE> Final_CPU_Out(M * K1);

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        float power_start = get_power_in_watts();
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        std::vector<OUT_DATATYPE> cpu_out1(M * N1);
        cpu_matmul(InVec, W1Vec, cpu_out1, M, K1, N1);
        std::vector<IN_DATATYPE> cpu_in2(M * N1);
        for(int i=0; i < M * N1; ++i) cpu_in2[i] = static_cast<IN_DATATYPE>(std::clamp(cpu_out1[i], (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
        std::vector<OUT_DATATYPE> cpu_out2(M * N2);
        cpu_matmul(cpu_in2, W2Vec, cpu_out2, M, N1, N2);
        std::vector<IN_DATATYPE> cpu_in3(M * N2);
        for(int i=0; i < M * N2; ++i) cpu_in3[i] = static_cast<IN_DATATYPE>(std::clamp(cpu_out2[i], (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
        cpu_matmul(cpu_in3, W3Vec, Final_CPU_Out, M, N2, K1);
        
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        float power_stop = get_power_in_watts();

        if (iter >= n_warmup_iterations) {
            cpu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
            cpu_power_total += (power_start + power_stop) / 2.0;
        }
    }

    // --- FINAL RESULTS ---
    std::cout << "\n--- Final Performance ---" << std::endl;
    std::cout << "Avg NPU Pipeline time: " << (npu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    std::cout << "Avg CPU Pipeline time: " << (cpu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    std::cout << "Avg NPU Pipeline Power: " << (npu_power_total / n_iterations) << " W." << std::endl;
    std::cout << "Avg CPU Pipeline Power: " << (cpu_power_total / n_iterations) << " W." << std::endl;

    // --- FINAL VERIFICATION ---
    int errors = 0;
    if(do_verify) {
        for(int i=0; i< M * K1; ++i) {
            if(Final_NPU_Out[i] != Final_CPU_Out[i]) {
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
