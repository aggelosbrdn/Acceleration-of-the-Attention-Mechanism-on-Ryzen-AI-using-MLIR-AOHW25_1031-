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
#include <thread> // Required for parallel execution
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
    if (!pipe) return -1.0;
    while (fgets(buffer, sizeof(buffer), pipe) != NULL) result += buffer;
    pclose(pipe);

    std::stringstream ss(result);
    std::string temp;
    float power = -1.0;
    while (ss >> temp) {
        if (std::stringstream(temp) >> power) return power;
    }
    return -1.0;
}


// Worker function for each parallel NPU thread
void run_matmul_on_npu(
    xrt::kernel& kernel,
    xrt::bo& bo_instr,
    const std::vector<uint32_t>& instr_v,
    xrt::bo& bo_in_a,
    xrt::bo& bo_in_b,
    xrt::bo& bo_out,
    const std::vector<IN_DATATYPE>& vec_a,
    const std::vector<IN_DATATYPE>& vec_b,
    std::vector<OUT_DATATYPE>& vec_out)
{
    bo_in_a.write(vec_a.data());
    bo_in_b.write(vec_b.data());
    bo_in_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto run = kernel(3, bo_instr, instr_v.size(), bo_in_a, bo_in_b, bo_out);
    run.wait();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out.read(vec_out.data());
}


int main(int argc, const char *argv[]) {
    cxxopts::Options options("Parallel MatMul Test");
    options.add_options()
        ("xclbin", "XCLBIN File", cxxopts::value<std::string>())
        ("insts", "Instruction File", cxxopts::value<std::string>())
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
    const int K = K_DIM;
    const int N = N_DIM;

    srand(12345);

    std::vector<IN_DATATYPE> A1(M * K), B1(K * N);
    std::vector<IN_DATATYPE> A2(M * K), B2(K * N);
    std::vector<IN_DATATYPE> A3(M * K), B3(K * N);
    
    initialize_vector(A1); initialize_vector(B1);
    initialize_vector(A2); initialize_vector(B2);
    initialize_vector(A3); initialize_vector(B3);

    std::vector<OUT_DATATYPE> C1_NPU(M * N), C2_NPU(M * N), C3_NPU(M * N);

    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
    std::vector<uint32_t> instr_v = test_utils::load_instr_binary(vm["insts"].as<std::string>());
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, vm["kernel"].as<std::string>());
    
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    bo_instr.write(instr_v.data());
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto bo_a1 = xrt::bo(device, A1.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b1 = xrt::bo(device, B1.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c1 = xrt::bo(device, C1_NPU.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    auto bo_a2 = xrt::bo(device, A2.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b2 = xrt::bo(device, B2.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c2 = xrt::bo(device, C2_NPU.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    auto bo_a3 = xrt::bo(device, A3.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_b3 = xrt::bo(device, B3.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_c3 = xrt::bo(device, C3_NPU.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    float npu_time_total = 0;
    // Add variable for NPU energy measurement
    float npu_power_total = 0;
    unsigned num_iter = n_iterations + n_warmup_iterations;

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        float power_start = get_power_in_watts();
        auto start = std::chrono::high_resolution_clock::now();

        std::thread t1(run_matmul_on_npu, std::ref(kernel), std::ref(bo_instr), std::ref(instr_v), std::ref(bo_a1), std::ref(bo_b1), std::ref(bo_c1), std::ref(A1), std::ref(B1), std::ref(C1_NPU));
        std::thread t2(run_matmul_on_npu, std::ref(kernel), std::ref(bo_instr), std::ref(instr_v), std::ref(bo_a2), std::ref(bo_b2), std::ref(bo_c2), std::ref(A2), std::ref(B2), std::ref(C2_NPU));
        std::thread t3(run_matmul_on_npu, std::ref(kernel), std::ref(bo_instr), std::ref(instr_v), std::ref(bo_a3), std::ref(bo_b3), std::ref(bo_c3), std::ref(A3), std::ref(B3), std::ref(C3_NPU));

        t1.join();
        t2.join();
        t3.join();

        auto stop = std::chrono::high_resolution_clock::now();
        float power_stop = get_power_in_watts();

        if (iter >= n_warmup_iterations) {
            npu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            npu_power_total += (power_start + power_stop) / 2.0;
        }
    }

    // --- SEQUENTIAL CPU TIMING BLOCK ---
    std::cout << "\n--- Running CPU reference for timing comparison ---" << std::endl;
    float cpu_time_total = 0;
    // Add variable for CPU energy measurement
    float cpu_power_total = 0;
    std::vector<OUT_DATATYPE> C1_CPU(M * N), C2_CPU(M * N), C3_CPU(M * N);

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        float power_start = get_power_in_watts();
        auto cpu_start = std::chrono::high_resolution_clock::now();
        
        cpu_matmul(A1, B1, C1_CPU, M, K, N);
        cpu_matmul(A2, B2, C2_CPU, M, K, N);
        cpu_matmul(A3, B3, C3_CPU, M, K, N);
        
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        float power_stop = get_power_in_watts();

        if (iter >= n_warmup_iterations) {
            cpu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
            cpu_power_total += (power_start + power_stop) / 2.0;
        }
    }

    // --- FINAL RESULTS ---
    std::cout << "\n--- Final Performance ---" << std::endl;
    std::cout << "Avg NPU Parallel Pipeline time: " << (npu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    std::cout << "Avg CPU Sequential Pipeline time: " << (cpu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    // ### MODIFICATION: Add new print statements for power ###
    std::cout << "Avg NPU Pipeline Power: " << (npu_power_total / n_iterations) << " W." << std::endl;
    std::cout << "Avg CPU Pipeline Power: " << (cpu_power_total / n_iterations) << " W." << std::endl;


    // --- FINAL VERIFICATION ---
    int errors = 0;
    if(do_verify) {
        for(int i=0; i< M * N; ++i) {
            if(C1_NPU[i] != C1_CPU[i]) errors++;
            if(C2_NPU[i] != C2_CPU[i]) errors++;
            if(C3_NPU[i] != C3_CPU[i]) errors++;
        }
    }

    if (!errors) {
        std::cout << "\nPASS! (NPU results match CPU reference)\n\n";
        return 0;
    } else {
        std::cout << "\nFailed. Error count: " << errors << "\n\n";
        return 1;
    }
}

