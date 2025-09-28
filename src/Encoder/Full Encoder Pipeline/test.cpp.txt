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
#include <thread>
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

// ### MODIFICATION: Helper function to get power consumption ###
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


// Worker function for the parallel Q, K, V generation
void run_matmul_on_npu(
    xrt::kernel& kernel, xrt::bo& bo_instr, const std::vector<uint32_t>& instr_v,
    xrt::bo& bo_in_a, xrt::bo& bo_in_b, xrt::bo& bo_out,
    const std::vector<IN_DATATYPE>& vec_a, const std::vector<IN_DATATYPE>& vec_b,
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

// CPU function for Layer Normalization
void cpu_layer_norm(std::vector<OUT_DATATYPE>& x, int M, int K) {
    float eps = 1e-5;
    for (int i = 0; i < M; ++i) {
        double sum = 0.0;
        for (int j = 0; j < K; ++j) sum += x[i * K + j];
        float mean = sum / K;

        double sq_sum = 0.0;
        for (int j = 0; j < K; ++j) sq_sum += (x[i * K + j] - mean) * (x[i * K + j] - mean);
        float variance = sq_sum / K;
        float std_dev = sqrt(variance + eps);

        for (int j = 0; j < K; ++j) x[i * K + j] = (x[i * K + j] - mean) / std_dev;
    }
}


int main(int argc, const char *argv[]) {
    // --- Command Line Argument Parsing ---
    cxxopts::Options options("Full Encoder Test");
    options.add_options()
        ("xclbin_qkv", "XCLBIN for QKV MatMul", cxxopts::value<std::string>())
        ("insts_qkv", "Instructions for QKV MatMul", cxxopts::value<std::string>())
        ("xclbin_sdpa1", "XCLBIN for SDPA MatMul1", cxxopts::value<std::string>())
        ("insts_sdpa1", "Instructions for SDPA MatMul1", cxxopts::value<std::string>())
        ("xclbin_sdpa2", "XCLBIN for SDPA MatMul2", cxxopts::value<std::string>())
        ("insts_sdpa2", "Instructions for SDPA MatMul2", cxxopts::value<std::string>())
        ("xclbin_ffn1", "XCLBIN for FFN MatMul1", cxxopts::value<std::string>())
        ("insts_ffn1", "Instructions for FFN MatMul1", cxxopts::value<std::string>())
        ("xclbin_ffn2", "XCLBIN for FFN MatMul2", cxxopts::value<std::string>())
        ("insts_ffn2", "Instructions for FFN MatMul2", cxxopts::value<std::string>())
        ("k,kernel", "Kernel Name", cxxopts::value<std::string>()->default_value("MLIR_AIE"))
        ("v,verify", "Enable verification", cxxopts::value<bool>()->default_value("true"))
        ("iters", "Number of iterations", cxxopts::value<int>()->default_value("10"))
        ("warmup", "Number of warmup iterations", cxxopts::value<int>()->default_value("5"));
    
    auto vm = options.parse(argc, argv);

    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();

    // --- Encoder Dimensions (passed from Makefile) ---
    const int M = M_DIM;
    const int K_MODEL = K_MODEL_DIM;
    const int K_HEAD = K_MODEL; // Simplified for this test
    const int FFN_HIDDEN = K_MODEL * 4;

    srand(12345);

    // --- Vector Initialization ---
    std::vector<IN_DATATYPE> X_in(M * K_MODEL);
    std::vector<IN_DATATYPE> W_Q(K_MODEL * K_HEAD), W_K(K_MODEL * K_HEAD), W_V(K_MODEL * K_HEAD);
    std::vector<IN_DATATYPE> FFN_W1(K_MODEL * FFN_HIDDEN), FFN_B1(FFN_HIDDEN);
    std::vector<IN_DATATYPE> FFN_W2(FFN_HIDDEN * K_MODEL), FFN_B2(K_MODEL);

    initialize_vector(X_in);
    initialize_vector(W_Q); initialize_vector(W_K); initialize_vector(W_V);
    initialize_vector(FFN_W1); initialize_vector(FFN_B1);
    initialize_vector(FFN_W2); initialize_vector(FFN_B2);

    // --- XRT Setup ---
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);

    auto xclbin_qkv = xrt::xclbin(vm["xclbin_qkv"].as<std::string>());
    std::vector<uint32_t> instr_qkv = test_utils::load_instr_binary(vm["insts_qkv"].as<std::string>());
    device.register_xclbin(xclbin_qkv);
    xrt::hw_context context_qkv(device, xclbin_qkv.get_uuid());
    auto kernel_qkv = xrt::kernel(context_qkv, vm["kernel"].as<std::string>());

    auto xclbin_sdpa1 = xrt::xclbin(vm["xclbin_sdpa1"].as<std::string>());
    std::vector<uint32_t> instr_sdpa1 = test_utils::load_instr_binary(vm["insts_sdpa1"].as<std::string>());
    device.register_xclbin(xclbin_sdpa1);
    xrt::hw_context context_sdpa1(device, xclbin_sdpa1.get_uuid());
    auto kernel_sdpa1 = xrt::kernel(context_sdpa1, vm["kernel"].as<std::string>());

    auto xclbin_sdpa2 = xrt::xclbin(vm["xclbin_sdpa2"].as<std::string>());
    std::vector<uint32_t> instr_sdpa2 = test_utils::load_instr_binary(vm["insts_sdpa2"].as<std::string>());
    device.register_xclbin(xclbin_sdpa2);
    xrt::hw_context context_sdpa2(device, xclbin_sdpa2.get_uuid());
    auto kernel_sdpa2 = xrt::kernel(context_sdpa2, vm["kernel"].as<std::string>());

    auto xclbin_ffn1 = xrt::xclbin(vm["xclbin_ffn1"].as<std::string>());
    std::vector<uint32_t> instr_ffn1 = test_utils::load_instr_binary(vm["insts_ffn1"].as<std::string>());
    device.register_xclbin(xclbin_ffn1);
    xrt::hw_context context_ffn1(device, xclbin_ffn1.get_uuid());
    auto kernel_ffn1 = xrt::kernel(context_ffn1, vm["kernel"].as<std::string>());

    auto xclbin_ffn2 = xrt::xclbin(vm["xclbin_ffn2"].as<std::string>());
    std::vector<uint32_t> instr_ffn2 = test_utils::load_instr_binary(vm["insts_ffn2"].as<std::string>());
    device.register_xclbin(xclbin_ffn2);
    xrt::hw_context context_ffn2(device, xclbin_ffn2.get_uuid());
    auto kernel_ffn2 = xrt::kernel(context_ffn2, vm["kernel"].as<std::string>());


    // --- NPU PIPELINE TIMING ---
    float npu_time_total = 0;
    // ### MODIFICATION: Add variable for NPU energy measurement ###
    float npu_power_total = 0;
    unsigned num_iter = n_iterations + n_warmup_iterations;
    std::vector<OUT_DATATYPE> Final_NPU_Out(M * K_MODEL);

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        // ### MODIFICATION: Get start power ###
        float power_start = get_power_in_watts();
        auto start = std::chrono::high_resolution_clock::now();

        // === STEP 1: Create Q, K, and V (Parallel NPU) ===
        std::vector<OUT_DATATYPE> Q_npu_32bit(M * K_HEAD), K_npu_32bit(M * K_HEAD), V_npu_32bit(M * K_HEAD);
        
        auto bo_instr_qkv = xrt::bo(device, instr_qkv.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_qkv.group_id(1));
        bo_instr_qkv.write(instr_qkv.data());
        bo_instr_qkv.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto bo_X_q = xrt::bo(device, X_in.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(3));
        auto bo_W_q = xrt::bo(device, W_Q.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(4));
        auto bo_Q_out = xrt::bo(device, Q_npu_32bit.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(5));

        auto bo_X_k = xrt::bo(device, X_in.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(3));
        auto bo_W_k = xrt::bo(device, W_K.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(4));
        auto bo_K_out = xrt::bo(device, K_npu_32bit.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(5));

        auto bo_X_v = xrt::bo(device, X_in.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(3));
        auto bo_W_v = xrt::bo(device, W_V.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(4));
        auto bo_V_out = xrt::bo(device, V_npu_32bit.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_qkv.group_id(5));

        std::thread t_q(run_matmul_on_npu, std::ref(kernel_qkv), std::ref(bo_instr_qkv), std::ref(instr_qkv), std::ref(bo_X_q), std::ref(bo_W_q), std::ref(bo_Q_out), std::ref(X_in), std::ref(W_Q), std::ref(Q_npu_32bit));
        std::thread t_k(run_matmul_on_npu, std::ref(kernel_qkv), std::ref(bo_instr_qkv), std::ref(instr_qkv), std::ref(bo_X_k), std::ref(bo_W_k), std::ref(bo_K_out), std::ref(X_in), std::ref(W_K), std::ref(K_npu_32bit));
        std::thread t_v(run_matmul_on_npu, std::ref(kernel_qkv), std::ref(bo_instr_qkv), std::ref(instr_qkv), std::ref(bo_X_v), std::ref(bo_W_v), std::ref(bo_V_out), std::ref(X_in), std::ref(W_V), std::ref(V_npu_32bit));

        t_q.join();
        t_k.join();
        t_v.join();

        std::vector<IN_DATATYPE> Q_npu_16bit(M * K_HEAD), K_npu_16bit(M * K_HEAD), V_npu_16bit(M * K_HEAD);
        for(int i=0; i<M*K_HEAD; ++i) {
            Q_npu_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)Q_npu_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
            K_npu_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)K_npu_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
            V_npu_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)V_npu_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
        }
        
        // === STEP 2: Scaled Dot-Product Attention (Hybrid) ===
        std::vector<OUT_DATATYPE> SDPA_Out(M * K_HEAD);
        
        std::vector<OUT_DATATYPE> MatMul1_sdpa_out(M * M);
        auto bo_instr_sdpa1 = xrt::bo(device, instr_sdpa1.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_sdpa1.group_id(1));
        auto bo_q_sdpa = xrt::bo(device, Q_npu_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_sdpa1.group_id(3));
        auto bo_k_sdpa = xrt::bo(device, K_npu_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_sdpa1.group_id(4));
        auto bo_out_sdpa1 = xrt::bo(device, MatMul1_sdpa_out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_sdpa1.group_id(5));
        
        bo_instr_sdpa1.write(instr_sdpa1.data());
        bo_q_sdpa.write(Q_npu_16bit.data());
        bo_k_sdpa.write(K_npu_16bit.data());
        bo_instr_sdpa1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_q_sdpa.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_k_sdpa.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run_sdpa1 = kernel_sdpa1(3, bo_instr_sdpa1, instr_sdpa1.size(), bo_q_sdpa, bo_k_sdpa, bo_out_sdpa1);
        run_sdpa1.wait();
        bo_out_sdpa1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out_sdpa1.read(MatMul1_sdpa_out.data());

        std::vector<IN_DATATYPE> Softmax_Out_16bit(M * M);
        float scale = 1.0f / sqrt(K_HEAD);
        for(int i = 0; i < M; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M; ++j) max_val = std::max(max_val, static_cast<float>(MatMul1_sdpa_out[i * M + j]));
            float sum_exp = 0.0f;
            for(int j = 0; j < M; ++j) sum_exp += expf((static_cast<float>(MatMul1_sdpa_out[i * M + j]) * scale) - max_val);
            for(int j = 0; j < M; ++j) {
                float softmax_val = expf((static_cast<float>(MatMul1_sdpa_out[i * M + j]) * scale) - max_val) / sum_exp;
                Softmax_Out_16bit[i * M + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
            }
        }

        auto bo_instr_sdpa2 = xrt::bo(device, instr_sdpa2.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_sdpa2.group_id(1));
        auto bo_sm_out = xrt::bo(device, Softmax_Out_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_sdpa2.group_id(3));
        auto bo_v_sdpa = xrt::bo(device, V_npu_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_sdpa2.group_id(4));
        auto bo_out_sdpa2 = xrt::bo(device, SDPA_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_sdpa2.group_id(5));

        bo_instr_sdpa2.write(instr_sdpa2.data());
        bo_sm_out.write(Softmax_Out_16bit.data());
        bo_v_sdpa.write(V_npu_16bit.data());
        bo_instr_sdpa2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_sm_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_v_sdpa.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run_sdpa2 = kernel_sdpa2(3, bo_instr_sdpa2, instr_sdpa2.size(), bo_sm_out, bo_v_sdpa, bo_out_sdpa2);
        run_sdpa2.wait();
        bo_out_sdpa2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out_sdpa2.read(SDPA_Out.data());


        // === STEP 3: First Add & Norm (CPU) ===
        std::vector<OUT_DATATYPE> AddNorm1_Out(M * K_MODEL);
        std::vector<IN_DATATYPE> X_in_16bit_for_add(M * K_MODEL);
        for(int i=0; i<M*K_MODEL; ++i) X_in_16bit_for_add[i] = X_in[i];
        
        for(int i=0; i<M*K_MODEL; ++i) { AddNorm1_Out[i] = SDPA_Out[i] + X_in_16bit_for_add[i]; }
        cpu_layer_norm(AddNorm1_Out, M, K_MODEL);

        // === STEP 4: Feed-Forward Network (Hybrid) ===
        std::vector<OUT_DATATYPE> FFN_Out(M * K_MODEL);
        
        std::vector<IN_DATATYPE> FFN_In_16bit(M * K_MODEL);
        for(int i=0; i<M*K_MODEL; ++i) FFN_In_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)AddNorm1_Out[i], (long)INT16_MIN, (long)INT16_MAX));

        std::vector<OUT_DATATYPE> FFN_MatMul1_Out(M * FFN_HIDDEN);
        auto bo_instr_ffn1 = xrt::bo(device, instr_ffn1.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_ffn1.group_id(1));
        auto bo_in_ffn1 = xrt::bo(device, FFN_In_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_ffn1.group_id(3));
        auto bo_w_ffn1 = xrt::bo(device, FFN_W1.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_ffn1.group_id(4));
        auto bo_out_ffn1 = xrt::bo(device, FFN_MatMul1_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_ffn1.group_id(5));

        bo_instr_ffn1.write(instr_ffn1.data());
        bo_in_ffn1.write(FFN_In_16bit.data());
        bo_w_ffn1.write(FFN_W1.data());
        bo_instr_ffn1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_in_ffn1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_w_ffn1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run_ffn1 = kernel_ffn1(3, bo_instr_ffn1, instr_ffn1.size(), bo_in_ffn1, bo_w_ffn1, bo_out_ffn1);
        run_ffn1.wait();
        bo_out_ffn1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out_ffn1.read(FFN_MatMul1_Out.data());

        std::vector<IN_DATATYPE> FFN_ReLU_Out_16bit(M * FFN_HIDDEN);
        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < FFN_HIDDEN; ++j) {
                int index = i * FFN_HIDDEN + j;
                ACC_DATATYPE biased_val_64 = (ACC_DATATYPE)FFN_MatMul1_Out[index] + FFN_B1[j];
                OUT_DATATYPE relu_32bit = std::max((OUT_DATATYPE)0, (OUT_DATATYPE)biased_val_64);
                FFN_ReLU_Out_16bit[index] = static_cast<IN_DATATYPE>(std::clamp(relu_32bit, (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
            }
        }

        auto bo_instr_ffn2 = xrt::bo(device, instr_ffn2.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_ffn2.group_id(1));
        auto bo_in_ffn2 = xrt::bo(device, FFN_ReLU_Out_16bit.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_ffn2.group_id(3));
        auto bo_w_ffn2 = xrt::bo(device, FFN_W2.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_ffn2.group_id(4));
        auto bo_out_ffn2 = xrt::bo(device, FFN_Out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel_ffn2.group_id(5));

        bo_instr_ffn2.write(instr_ffn2.data());
        bo_in_ffn2.write(FFN_ReLU_Out_16bit.data());
        bo_w_ffn2.write(FFN_W2.data());
        bo_instr_ffn2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_in_ffn2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_w_ffn2.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run_ffn2 = kernel_ffn2(3, bo_instr_ffn2, instr_ffn2.size(), bo_in_ffn2, bo_w_ffn2, bo_out_ffn2);
        run_ffn2.wait();
        bo_out_ffn2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out_ffn2.read(FFN_Out.data());

        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < K_MODEL; ++j) {
                FFN_Out[i * K_MODEL + j] += FFN_B2[j];
            }
        }

        // === STEP 5: Second Add & Norm (CPU) ===
        for(int i=0; i<M*K_MODEL; ++i) { Final_NPU_Out[i] = FFN_Out[i] + AddNorm1_Out[i]; }
        cpu_layer_norm(Final_NPU_Out, M, K_MODEL);
        
        auto stop = std::chrono::high_resolution_clock::now();
        // ### MODIFICATION: Get stop power ###
        float power_stop = get_power_in_watts();

        if (iter >= n_warmup_iterations) {
            npu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            // ### MODIFICATION: Accumulate power reading ###
            npu_power_total += (power_start + power_stop) / 2.0;
        }
    }

    // --- PURE CPU TIMING BLOCK ---
    std::cout << "\n--- Running CPU reference for timing comparison ---" << std::endl;
    float cpu_time_total = 0;
    // ### MODIFICATION: Add variable for CPU energy measurement ###
    float cpu_power_total = 0;
    std::vector<OUT_DATATYPE> Final_CPU_Out(M * K_MODEL);

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        // ### MODIFICATION: Get start power ###
        float power_start = get_power_in_watts();
        auto cpu_start = std::chrono::high_resolution_clock::now();
        // === STEP 1: Create Q, K, and V (Sequential CPU) ===
        std::vector<OUT_DATATYPE> Q_cpu(M * K_HEAD), K_cpu(M * K_HEAD), V_cpu(M * K_HEAD);
        
        // Helper lambda for CPU matmul
        auto cpu_matmul = [&](const auto& A, const auto& B, auto& C, int m, int k, int n) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    ACC_DATATYPE acc = 0;
                    for (int l = 0; l < k; ++l) {
                        acc += (ACC_DATATYPE)A[i * k + l] * (ACC_DATATYPE)B[l * n + j];
                    }
                    C[i * n + j] = (OUT_DATATYPE)acc;
                }
            }
        };

        cpu_matmul(X_in, W_Q, Q_cpu, M, K_MODEL, K_HEAD);
        cpu_matmul(X_in, W_K, K_cpu, M, K_MODEL, K_HEAD);
        cpu_matmul(X_in, W_V, V_cpu, M, K_MODEL, K_HEAD);

        std::vector<IN_DATATYPE> Q_cpu_16bit(M * K_HEAD), K_cpu_16bit(M * K_HEAD), V_cpu_16bit(M * K_HEAD);
        for(int i=0; i<M*K_HEAD; ++i) {
            Q_cpu_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)Q_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
            K_cpu_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)K_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
            V_cpu_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)V_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
        }

        // === STEP 2: Scaled Dot-Product Attention (CPU) ===
        std::vector<OUT_DATATYPE> SDPA_Out_cpu(M * K_HEAD);
        std::vector<OUT_DATATYPE> cpu_matmul1_sdpa_out(M * M);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                ACC_DATATYPE acc = 0;
                for (int k = 0; k < K_HEAD; ++k) {
                    acc += (ACC_DATATYPE)Q_cpu_16bit[i * K_HEAD + k] * (ACC_DATATYPE)K_cpu_16bit[j * K_HEAD + k];
                }
                cpu_matmul1_sdpa_out[i * M + j] = (OUT_DATATYPE)acc;
            }
        }

        std::vector<IN_DATATYPE> cpu_softmax_out_16bit(M * M);
        float scale = 1.0f / sqrt(K_HEAD);
        for(int i = 0; i < M; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M; ++j) max_val = std::max(max_val, static_cast<float>(cpu_matmul1_sdpa_out[i * M + j]));
            float sum_exp = 0.0f;
            for(int j = 0; j < M; ++j) sum_exp += expf((static_cast<float>(cpu_matmul1_sdpa_out[i * M + j]) * scale) - max_val);
            for(int j = 0; j < M; ++j) {
                float softmax_val = expf((static_cast<float>(cpu_matmul1_sdpa_out[i * M + j]) * scale) - max_val) / sum_exp;
                cpu_softmax_out_16bit[i * M + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
            }
        }
        cpu_matmul(cpu_softmax_out_16bit, V_cpu_16bit, SDPA_Out_cpu, M, M, K_HEAD);

        // === STEP 3: First Add & Norm (CPU) ===
        std::vector<OUT_DATATYPE> AddNorm1_Out_cpu(M * K_MODEL);
        std::vector<OUT_DATATYPE> X_in_32bit_for_add(M * K_MODEL);
        for(int i=0; i<M*K_MODEL; ++i) X_in_32bit_for_add[i] = static_cast<OUT_DATATYPE>(X_in[i]);
        for(int i=0; i<M*K_MODEL; ++i) { AddNorm1_Out_cpu[i] = SDPA_Out_cpu[i] + X_in_32bit_for_add[i]; }
        cpu_layer_norm(AddNorm1_Out_cpu, M, K_MODEL);

        // === STEP 4: Feed-Forward Network (CPU) ===
        std::vector<IN_DATATYPE> FFN_In_16bit_cpu(M * K_MODEL);
        for(int i=0; i<M*K_MODEL; ++i) FFN_In_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)AddNorm1_Out_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
        
        std::vector<OUT_DATATYPE> FFN_MatMul1_Out_cpu(M * FFN_HIDDEN);
        cpu_matmul(FFN_In_16bit_cpu, FFN_W1, FFN_MatMul1_Out_cpu, M, K_MODEL, FFN_HIDDEN);

        std::vector<IN_DATATYPE> FFN_ReLU_Out_16bit_cpu(M * FFN_HIDDEN);
        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < FFN_HIDDEN; ++j) {
                int index = i * FFN_HIDDEN + j;
                ACC_DATATYPE biased_val_64 = (ACC_DATATYPE)FFN_MatMul1_Out_cpu[index] + FFN_B1[j];
                OUT_DATATYPE relu_32bit = std::max((OUT_DATATYPE)0, (OUT_DATATYPE)biased_val_64);
                FFN_ReLU_Out_16bit_cpu[index] = static_cast<IN_DATATYPE>(std::clamp(relu_32bit, (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
            }
        }
        
        std::vector<OUT_DATATYPE> FFN_Out_cpu(M * K_MODEL);
        cpu_matmul(FFN_ReLU_Out_16bit_cpu, FFN_W2, FFN_Out_cpu, M, FFN_HIDDEN, K_MODEL);

        for(int i = 0; i < M; ++i) {
            for(int j = 0; j < K_MODEL; ++j) {
                FFN_Out_cpu[i * K_MODEL + j] += FFN_B2[j];
            }
        }

        // === STEP 5: Second Add & Norm (CPU) ===
        for(int i=0; i<M*K_MODEL; ++i) { Final_CPU_Out[i] = FFN_Out_cpu[i] + AddNorm1_Out_cpu[i]; }
        cpu_layer_norm(Final_CPU_Out, M, K_MODEL);
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        // ### MODIFICATION: Get stop power ###
        float power_stop = get_power_in_watts();

        if (iter >= n_warmup_iterations) {
            cpu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
            // ### MODIFICATION: Accumulate power reading ###
            cpu_power_total += (power_start + power_stop) / 2.0;
        }
    }

    // --- FINAL RESULTS ---
    std::cout << "\n--- Final Performance ---" << std::endl;
    std::cout << "Avg NPU Pipeline time: " << (npu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    std::cout << "Avg CPU Pipeline time: " << (cpu_time_total / n_iterations) / 1000.0 << "ms." << std::endl;
    // ### MODIFICATION: Add new print statements for power ###
    std::cout << "Avg NPU Pipeline Power: " << (npu_power_total / n_iterations) << " W." << std::endl;
    std::cout << "Avg CPU Pipeline Power: " << (cpu_power_total / n_iterations) << " W." << std::endl;


    // --- FINAL VERIFICATION ---
    int errors = 0;
    if(do_verify) {
        for(int i=0; i< M * K_MODEL; ++i) {
            if(Final_NPU_Out[i] != Final_CPU_Out[i]) {
                if(errors < 10) { // Print first 10 errors to avoid flooding the console
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
