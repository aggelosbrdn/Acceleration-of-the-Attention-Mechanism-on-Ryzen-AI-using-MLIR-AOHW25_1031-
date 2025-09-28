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

#define DTYPE_IN int16_t
#define DTYPE_OUT int32_t
#define DTYPE_ACC int64_t

using IN_DATATYPE = DTYPE_IN;
using OUT_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;

template <typename T>
void initialize_vector(std::vector<T>& vec) {
    for (auto& val : vec) {
        val = static_cast<T>((rand() % 20) - 10);
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


void run_npu_matmul(
    xrt::kernel& kernel, const std::vector<uint32_t>& instr_v,
    const std::vector<IN_DATATYPE>& vec_a, const std::vector<IN_DATATYPE>& vec_b,
    std::vector<OUT_DATATYPE>& vec_out, xrt::device& device)
{
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_in_a = xrt::bo(device, vec_a.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_in_b = xrt::bo(device, vec_b.size() * sizeof(IN_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out = xrt::bo(device, vec_out.size() * sizeof(OUT_DATATYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    bo_instr.write(instr_v.data());
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    bo_in_a.write(vec_a.data());
    bo_in_b.write(vec_b.data());
    bo_in_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(3, bo_instr, instr_v.size(), bo_in_a, bo_in_b, bo_out);
    run.wait();

    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out.read(vec_out.data());
}


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
        for (int j = 0; j < K; ++j) x[i * K + j] = (OUT_DATATYPE)((x[i * K + j] - mean) / std_dev);
    }
}

int main(int argc, const char *argv[]) {
    cxxopts::Options options("Full Decoder Test");
    options.add_options()
        ("xclbin_sa_qkv", "XCLBIN for Self-Attention QKV", cxxopts::value<std::string>())
        ("insts_sa_qkv", "Instructions for Self-Attention QKV", cxxopts::value<std::string>())
        ("xclbin_sa_sdpa1", "XCLBIN for Self-Attention SDPA1", cxxopts::value<std::string>())
        ("insts_sa_sdpa1", "Instructions for Self-Attention SDPA1", cxxopts::value<std::string>())
        ("xclbin_sa_sdpa2", "XCLBIN for Self-Attention SDPA2", cxxopts::value<std::string>())
        ("insts_sa_sdpa2", "Instructions for Self-Attention SDPA2", cxxopts::value<std::string>())
        ("xclbin_ca_q", "XCLBIN for Cross-Attention Q", cxxopts::value<std::string>())
        ("insts_ca_q", "Instructions for Cross-Attention Q", cxxopts::value<std::string>())
        ("xclbin_ca_kv", "XCLBIN for Cross-Attention KV", cxxopts::value<std::string>())
        ("insts_ca_kv", "Instructions for Cross-Attention KV", cxxopts::value<std::string>())
        ("xclbin_ca_sdpa1", "XCLBIN for Cross-Attention SDPA1", cxxopts::value<std::string>())
        ("insts_ca_sdpa1", "Instructions for Cross-Attention SDPA1", cxxopts::value<std::string>())
        ("xclbin_ca_sdpa2", "XCLBIN for Cross-Attention SDPA2", cxxopts::value<std::string>())
        ("insts_ca_sdpa2", "Instructions for Cross-Attention SDPA2", cxxopts::value<std::string>())
        ("xclbin_ffn1", "XCLBIN for FFN1", cxxopts::value<std::string>())
        ("insts_ffn1", "Instructions for FFN1", cxxopts::value<std::string>())
        ("xclbin_ffn2", "XCLBIN for FFN2", cxxopts::value<std::string>())
        ("insts_ffn2", "Instructions for FFN2", cxxopts::value<std::string>())
        ("k,kernel", "Kernel Name", cxxopts::value<std::string>()->default_value("MLIR_AIE"))
        ("v,verify", "Enable verification", cxxopts::value<bool>()->default_value("true"))
        ("iters", "Number of iterations", cxxopts::value<int>()->default_value("10"))
        ("warmup", "Number of warmup iterations", cxxopts::value<int>()->default_value("5"));
    
    auto vm = options.parse(argc, argv);

    int do_verify = vm["verify"].as<bool>();
    int n_iterations = vm["iters"].as<int>();
    int n_warmup_iterations = vm["warmup"].as<int>();
    unsigned num_iter = n_iterations + n_warmup_iterations;

    const int M_TGT = M_TGT_DIM;
    const int M_SRC = M_SRC_DIM;
    const int K_MODEL = K_MODEL_DIM;
    const int K_HEAD = K_MODEL; 
    const int FFN_HIDDEN = K_MODEL * 4;

    srand(12345);

    // --- Vector Initialization ---
    std::vector<IN_DATATYPE> Decoder_Input(M_TGT * K_MODEL);
    std::vector<IN_DATATYPE> Encoder_Output(M_SRC * K_MODEL);
    initialize_vector(Decoder_Input);
    initialize_vector(Encoder_Output);
    
    std::vector<IN_DATATYPE> W_Q_sa(K_MODEL * K_HEAD), W_K_sa(K_MODEL * K_HEAD), W_V_sa(K_MODEL * K_HEAD);
    initialize_vector(W_Q_sa); initialize_vector(W_K_sa); initialize_vector(W_V_sa);

    std::vector<IN_DATATYPE> W_Q_ca(K_MODEL * K_HEAD), W_K_ca(K_MODEL * K_HEAD), W_V_ca(K_MODEL * K_HEAD);
    initialize_vector(W_Q_ca); initialize_vector(W_K_ca); initialize_vector(W_V_ca);

    std::vector<IN_DATATYPE> FFN_W1(K_MODEL * FFN_HIDDEN), FFN_B1(FFN_HIDDEN);
    std::vector<IN_DATATYPE> FFN_W2(FFN_HIDDEN * K_MODEL), FFN_B2(K_MODEL);
    initialize_vector(FFN_W1); initialize_vector(FFN_B1);
    initialize_vector(FFN_W2); initialize_vector(FFN_B2);

    // --- XRT Setup ---
    unsigned int device_index = 0;
    auto device = xrt::device(device_index);
    
    auto xclbin_sa_qkv_handle = xrt::xclbin(vm["xclbin_sa_qkv"].as<std::string>());
    auto uuid = xclbin_sa_qkv_handle.get_uuid();
    device.register_xclbin(xclbin_sa_qkv_handle);
    
    xrt::hw_context context(device, uuid);

    auto load_kernel = [&](const std::string& xclbin_path, const std::string& insts_path) {
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        auto kernel = xrt::kernel(context, vm["kernel"].as<std::string>());
        auto insts = test_utils::load_instr_binary(insts_path);
        return std::make_tuple(kernel, insts);
    };

    auto [kernel_sa_qkv, instr_sa_qkv] = load_kernel(vm["xclbin_sa_qkv"].as<std::string>(), vm["insts_sa_qkv"].as<std::string>());
    auto [kernel_sa_sdpa1, instr_sa_sdpa1] = load_kernel(vm["xclbin_sa_sdpa1"].as<std::string>(), vm["insts_sa_sdpa1"].as<std::string>());
    auto [kernel_sa_sdpa2, instr_sa_sdpa2] = load_kernel(vm["xclbin_sa_sdpa2"].as<std::string>(), vm["insts_sa_sdpa2"].as<std::string>());
    auto [kernel_ca_q, instr_ca_q] = load_kernel(vm["xclbin_ca_q"].as<std::string>(), vm["insts_ca_q"].as<std::string>());
    auto [kernel_ca_kv, instr_ca_kv] = load_kernel(vm["xclbin_ca_kv"].as<std::string>(), vm["insts_ca_kv"].as<std::string>());
    auto [kernel_ca_sdpa1, instr_ca_sdpa1] = load_kernel(vm["xclbin_ca_sdpa1"].as<std::string>(), vm["insts_ca_sdpa1"].as<std::string>());
    auto [kernel_ca_sdpa2, instr_ca_sdpa2] = load_kernel(vm["xclbin_ca_sdpa2"].as<std::string>(), vm["insts_ca_sdpa2"].as<std::string>());
    auto [kernel_ffn1, instr_ffn1] = load_kernel(vm["xclbin_ffn1"].as<std::string>(), vm["insts_ffn1"].as<std::string>());
    auto [kernel_ffn2, instr_ffn2] = load_kernel(vm["xclbin_ffn2"].as<std::string>(), vm["insts_ffn2"].as<std::string>());

    // --- NPU PIPELINE ---
    float npu_time_total = 0;
    float npu_power_total = 0;
    std::vector<OUT_DATATYPE> Final_NPU_Out(M_TGT * K_MODEL);
    
    for (unsigned iter = 0; iter < num_iter; ++iter) {
        float power_start = get_power_in_watts();
        auto start = std::chrono::high_resolution_clock::now();

        // === STEP 1: Masked Self-Attention ===
        std::vector<OUT_DATATYPE> Q_sa_32bit(M_TGT * K_HEAD), K_sa_32bit(M_TGT * K_HEAD), V_sa_32bit(M_TGT * K_HEAD);
        
        std::thread t_sa_q([&]{ run_npu_matmul(kernel_sa_qkv, instr_sa_qkv, Decoder_Input, W_Q_sa, Q_sa_32bit, device); });
        std::thread t_sa_k([&]{ run_npu_matmul(kernel_sa_qkv, instr_sa_qkv, Decoder_Input, W_K_sa, K_sa_32bit, device); });
        std::thread t_sa_v([&]{ run_npu_matmul(kernel_sa_qkv, instr_sa_qkv, Decoder_Input, W_V_sa, V_sa_32bit, device); });
        t_sa_q.join(); t_sa_k.join(); t_sa_v.join();

        std::vector<IN_DATATYPE> Q_sa_16bit(M_TGT * K_HEAD), K_sa_16bit(M_TGT * K_HEAD), V_sa_16bit(M_TGT * K_HEAD);
        for(int i=0; i<M_TGT*K_HEAD; ++i) {
            Q_sa_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)Q_sa_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
            K_sa_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)K_sa_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
            V_sa_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)V_sa_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
        }

        std::vector<OUT_DATATYPE> MatMul1_sa_out(M_TGT * M_TGT);
        run_npu_matmul(kernel_sa_sdpa1, instr_sa_sdpa1, Q_sa_16bit, K_sa_16bit, MatMul1_sa_out, device);
        
        std::vector<IN_DATATYPE> Softmax_sa_Out_16bit(M_TGT * M_TGT);
        float scale_sa = 1.0f / sqrt(K_HEAD);
        for(int i = 0; i < M_TGT; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M_TGT; ++j) {
                if (j <= i) max_val = std::max(max_val, static_cast<float>(MatMul1_sa_out[i * M_TGT + j]));
            }
            float sum_exp = 0.0f;
            for(int j = 0; j < M_TGT; ++j) {
                if (j <= i) sum_exp += expf((static_cast<float>(MatMul1_sa_out[i * M_TGT + j]) * scale_sa) - max_val);
            }
            for(int j = 0; j < M_TGT; ++j) {
                if (j <= i) {
                    float softmax_val = expf((static_cast<float>(MatMul1_sa_out[i * M_TGT + j]) * scale_sa) - max_val) / sum_exp;
                    Softmax_sa_Out_16bit[i * M_TGT + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
                } else {
                    Softmax_sa_Out_16bit[i * M_TGT + j] = 0;
                }
            }
        }
        
        std::vector<OUT_DATATYPE> SA_Out(M_TGT * K_HEAD);
        run_npu_matmul(kernel_sa_sdpa2, instr_sa_sdpa2, Softmax_sa_Out_16bit, V_sa_16bit, SA_Out, device);

        // === STEP 2: Add & Norm 1 (CPU) ===
        std::vector<OUT_DATATYPE> AddNorm1_Out(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) { AddNorm1_Out[i] = SA_Out[i] + static_cast<OUT_DATATYPE>(Decoder_Input[i]); }
        cpu_layer_norm(AddNorm1_Out, M_TGT, K_MODEL);

        // === STEP 3: Cross-Attention ===
        std::vector<IN_DATATYPE> AddNorm1_Out_16bit(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) AddNorm1_Out_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)AddNorm1_Out[i], (long)INT16_MIN, (long)INT16_MAX));
        
        std::vector<OUT_DATATYPE> Q_ca_32bit(M_TGT * K_HEAD), K_ca_32bit(M_SRC * K_HEAD), V_ca_32bit(M_SRC * K_HEAD);
        
        run_npu_matmul(kernel_ca_q, instr_ca_q, AddNorm1_Out_16bit, W_Q_ca, Q_ca_32bit, device);
        std::thread t_ca_k([&]{ run_npu_matmul(kernel_ca_kv, instr_ca_kv, Encoder_Output, W_K_ca, K_ca_32bit, device); });
        std::thread t_ca_v([&]{ run_npu_matmul(kernel_ca_kv, instr_ca_kv, Encoder_Output, W_V_ca, V_ca_32bit, device); });
        t_ca_k.join(); t_ca_v.join();

        std::vector<IN_DATATYPE> Q_ca_16bit(M_TGT * K_HEAD), K_ca_16bit(M_SRC * K_HEAD), V_ca_16bit(M_SRC * K_HEAD);
        for(int i=0; i<M_TGT*K_HEAD; ++i) Q_ca_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)Q_ca_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
        for(int i=0; i<M_SRC*K_HEAD; ++i) {
            K_ca_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)K_ca_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
            V_ca_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)V_ca_32bit[i], (long)INT16_MIN, (long)INT16_MAX));
        }

        std::vector<OUT_DATATYPE> MatMul1_ca_out(M_TGT * M_SRC);
        run_npu_matmul(kernel_ca_sdpa1, instr_ca_sdpa1, Q_ca_16bit, K_ca_16bit, MatMul1_ca_out, device);

        std::vector<IN_DATATYPE> Softmax_ca_Out_16bit(M_TGT * M_SRC);
        float scale_ca = 1.0f / sqrt(K_HEAD);
        for(int i = 0; i < M_TGT; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M_SRC; ++j) max_val = std::max(max_val, static_cast<float>(MatMul1_ca_out[i * M_SRC + j]));
            float sum_exp = 0.0f;
            for(int j = 0; j < M_SRC; ++j) sum_exp += expf((static_cast<float>(MatMul1_ca_out[i * M_SRC + j]) * scale_ca) - max_val);
            for(int j = 0; j < M_SRC; ++j) {
                float softmax_val = expf((static_cast<float>(MatMul1_ca_out[i * M_SRC + j]) * scale_ca) - max_val) / sum_exp;
                Softmax_ca_Out_16bit[i * M_SRC + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
            }
        }
        
        std::vector<OUT_DATATYPE> CA_Out(M_TGT * K_HEAD);
        run_npu_matmul(kernel_ca_sdpa2, instr_ca_sdpa2, Softmax_ca_Out_16bit, V_ca_16bit, CA_Out, device);

        // === STEP 4: Add & Norm 2 (CPU) ===
        std::vector<OUT_DATATYPE> AddNorm2_Out(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) { AddNorm2_Out[i] = CA_Out[i] + AddNorm1_Out[i]; }
        cpu_layer_norm(AddNorm2_Out, M_TGT, K_MODEL);
        
        // === STEP 5: Feed-Forward Network ===
        std::vector<IN_DATATYPE> AddNorm2_Out_16bit(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) AddNorm2_Out_16bit[i] = static_cast<IN_DATATYPE>(std::clamp((long)AddNorm2_Out[i], (long)INT16_MIN, (long)INT16_MAX));
        
        std::vector<OUT_DATATYPE> FFN_MatMul1_Out(M_TGT * FFN_HIDDEN);
        run_npu_matmul(kernel_ffn1, instr_ffn1, AddNorm2_Out_16bit, FFN_W1, FFN_MatMul1_Out, device);

        std::vector<IN_DATATYPE> FFN_ReLU_Out_16bit(M_TGT * FFN_HIDDEN);
        for(int i = 0; i < M_TGT; ++i) {
            for(int j = 0; j < FFN_HIDDEN; ++j) {
                int index = i * FFN_HIDDEN + j;
                ACC_DATATYPE biased_val_64 = (ACC_DATATYPE)FFN_MatMul1_Out[index] + FFN_B1[j];
                OUT_DATATYPE relu_32bit = std::max((OUT_DATATYPE)0, (OUT_DATATYPE)biased_val_64);
                FFN_ReLU_Out_16bit[index] = static_cast<IN_DATATYPE>(std::clamp(relu_32bit, (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
            }
        }

        std::vector<OUT_DATATYPE> FFN_Out(M_TGT * K_MODEL);
        run_npu_matmul(kernel_ffn2, instr_ffn2, FFN_ReLU_Out_16bit, FFN_W2, FFN_Out, device);

        for(int i = 0; i < M_TGT; ++i) {
            for(int j = 0; j < K_MODEL; ++j) {
                FFN_Out[i * K_MODEL + j] += FFN_B2[j];
            }
        }
        
        // === STEP 6: Add & Norm 3 (CPU) ===
        for(int i=0; i<M_TGT*K_MODEL; ++i) { Final_NPU_Out[i] = FFN_Out[i] + AddNorm2_Out[i]; }
        cpu_layer_norm(Final_NPU_Out, M_TGT, K_MODEL);

        auto stop = std::chrono::high_resolution_clock::now();
        float power_stop = get_power_in_watts();
        if (iter >= n_warmup_iterations) {
            npu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            if (power_start > 0 && power_stop > 0) {
                npu_power_total += (power_start + power_stop) / 2.0;
            }
        }
    }

    // --- PURE CPU TIMING BLOCK ---
    std::cout << "\n--- Running CPU reference for verification ---" << std::endl;
    float cpu_time_total = 0;
    float cpu_power_total = 0;
    std::vector<OUT_DATATYPE> Final_CPU_Out(M_TGT * K_MODEL);

    for (unsigned iter = 0; iter < num_iter; ++iter) {
        float power_start = get_power_in_watts();
        auto cpu_start = std::chrono::high_resolution_clock::now();

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

        // === CPU STEP 1: Masked Self-Attention ===
        std::vector<OUT_DATATYPE> Q_sa_cpu(M_TGT * K_HEAD), K_sa_cpu(M_TGT * K_HEAD), V_sa_cpu(M_TGT * K_HEAD);
        cpu_matmul(Decoder_Input, W_Q_sa, Q_sa_cpu, M_TGT, K_MODEL, K_HEAD);
        cpu_matmul(Decoder_Input, W_K_sa, K_sa_cpu, M_TGT, K_MODEL, K_HEAD);
        cpu_matmul(Decoder_Input, W_V_sa, V_sa_cpu, M_TGT, K_MODEL, K_HEAD);

        std::vector<IN_DATATYPE> Q_sa_16bit_cpu(M_TGT * K_HEAD), K_sa_16bit_cpu(M_TGT * K_HEAD), V_sa_16bit_cpu(M_TGT * K_HEAD);
        for(int i=0; i<M_TGT*K_HEAD; ++i) {
            Q_sa_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)Q_sa_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
            K_sa_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)K_sa_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
            V_sa_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)V_sa_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
        }

        std::vector<OUT_DATATYPE> MatMul1_sa_out_cpu(M_TGT * M_TGT);
        for (int i = 0; i < M_TGT; ++i) {
            for (int j = 0; j < M_TGT; ++j) {
                ACC_DATATYPE acc = 0;
                for (int k = 0; k < K_HEAD; ++k) {
                    acc += (ACC_DATATYPE)Q_sa_16bit_cpu[i * K_HEAD + k] * (ACC_DATATYPE)K_sa_16bit_cpu[j * K_HEAD + k];
                }
                MatMul1_sa_out_cpu[i * M_TGT + j] = (OUT_DATATYPE)acc;
            }
        }

        std::vector<IN_DATATYPE> Softmax_sa_Out_16bit_cpu(M_TGT * M_TGT);
        float scale_sa = 1.0f / sqrt(K_HEAD);
        for(int i = 0; i < M_TGT; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M_TGT; ++j) {
                if (j <= i) max_val = std::max(max_val, static_cast<float>(MatMul1_sa_out_cpu[i * M_TGT + j]));
            }
            float sum_exp = 0.0f;
            for(int j = 0; j < M_TGT; ++j) {
                if (j <= i) sum_exp += expf((static_cast<float>(MatMul1_sa_out_cpu[i * M_TGT + j]) * scale_sa) - max_val);
            }
            for(int j = 0; j < M_TGT; ++j) {
                if (j <= i) {
                    float softmax_val = expf((static_cast<float>(MatMul1_sa_out_cpu[i * M_TGT + j]) * scale_sa) - max_val) / sum_exp;
                    Softmax_sa_Out_16bit_cpu[i * M_TGT + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
                } else {
                    Softmax_sa_Out_16bit_cpu[i * M_TGT + j] = 0;
                }
            }
        }
        
        std::vector<OUT_DATATYPE> SA_Out_cpu(M_TGT * K_HEAD);
        cpu_matmul(Softmax_sa_Out_16bit_cpu, V_sa_16bit_cpu, SA_Out_cpu, M_TGT, M_TGT, K_HEAD);

        // === CPU STEP 2: Add & Norm 1 ===
        std::vector<OUT_DATATYPE> AddNorm1_Out_cpu(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) { AddNorm1_Out_cpu[i] = SA_Out_cpu[i] + static_cast<OUT_DATATYPE>(Decoder_Input[i]); }
        cpu_layer_norm(AddNorm1_Out_cpu, M_TGT, K_MODEL);

        // === CPU STEP 3: Cross-Attention ===
        std::vector<IN_DATATYPE> AddNorm1_Out_16bit_cpu(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) AddNorm1_Out_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)AddNorm1_Out_cpu[i], (long)INT16_MIN, (long)INT16_MAX));

        std::vector<OUT_DATATYPE> Q_ca_cpu(M_TGT * K_HEAD), K_ca_cpu(M_SRC * K_HEAD), V_ca_cpu(M_SRC * K_HEAD);
        cpu_matmul(AddNorm1_Out_16bit_cpu, W_Q_ca, Q_ca_cpu, M_TGT, K_MODEL, K_HEAD);
        cpu_matmul(Encoder_Output, W_K_ca, K_ca_cpu, M_SRC, K_MODEL, K_HEAD);
        cpu_matmul(Encoder_Output, W_V_ca, V_ca_cpu, M_SRC, K_MODEL, K_HEAD);

        std::vector<IN_DATATYPE> Q_ca_16bit_cpu(M_TGT * K_HEAD), K_ca_16bit_cpu(M_SRC * K_HEAD), V_ca_16bit_cpu(M_SRC * K_HEAD);
        for(int i=0; i<M_TGT*K_HEAD; ++i) Q_ca_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)Q_ca_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
        for(int i=0; i<M_SRC*K_HEAD; ++i) {
            K_ca_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)K_ca_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
            V_ca_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)V_ca_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
        }

        std::vector<OUT_DATATYPE> MatMul1_ca_out_cpu(M_TGT * M_SRC);
        for (int i = 0; i < M_TGT; ++i) {
            for (int j = 0; j < M_SRC; ++j) {
                ACC_DATATYPE acc = 0;
                for (int k = 0; k < K_HEAD; ++k) {
                    acc += (ACC_DATATYPE)Q_ca_16bit_cpu[i * K_HEAD + k] * (ACC_DATATYPE)K_ca_16bit_cpu[j * K_HEAD + k];
                }
                MatMul1_ca_out_cpu[i * M_SRC + j] = (OUT_DATATYPE)acc;
            }
        }

        std::vector<IN_DATATYPE> Softmax_ca_Out_16bit_cpu(M_TGT * M_SRC);
        float scale_ca = 1.0f / sqrt(K_HEAD);
        for(int i = 0; i < M_TGT; ++i) {
            float max_val = -FLT_MAX;
            for(int j = 0; j < M_SRC; ++j) max_val = std::max(max_val, static_cast<float>(MatMul1_ca_out_cpu[i * M_SRC + j]));
            float sum_exp = 0.0f;
            for(int j = 0; j < M_SRC; ++j) sum_exp += expf((static_cast<float>(MatMul1_ca_out_cpu[i * M_SRC + j]) * scale_ca) - max_val);
            for(int j = 0; j < M_SRC; ++j) {
                float softmax_val = expf((static_cast<float>(MatMul1_ca_out_cpu[i * M_SRC + j]) * scale_ca) - max_val) / sum_exp;
                Softmax_ca_Out_16bit_cpu[i * M_SRC + j] = static_cast<IN_DATATYPE>(softmax_val * 127.0f);
            }
        }

        std::vector<OUT_DATATYPE> CA_Out_cpu(M_TGT * K_HEAD);
        cpu_matmul(Softmax_ca_Out_16bit_cpu, V_ca_16bit_cpu, CA_Out_cpu, M_TGT, M_SRC, K_HEAD);

        // === CPU STEP 4: Add & Norm 2 ===
        std::vector<OUT_DATATYPE> AddNorm2_Out_cpu(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) { AddNorm2_Out_cpu[i] = CA_Out_cpu[i] + AddNorm1_Out_cpu[i]; }
        cpu_layer_norm(AddNorm2_Out_cpu, M_TGT, K_MODEL);

        // === CPU STEP 5: Feed-Forward Network ===
        std::vector<IN_DATATYPE> FFN_In_16bit_cpu(M_TGT * K_MODEL);
        for(int i=0; i<M_TGT*K_MODEL; ++i) FFN_In_16bit_cpu[i] = static_cast<IN_DATATYPE>(std::clamp((long)AddNorm2_Out_cpu[i], (long)INT16_MIN, (long)INT16_MAX));
        
        std::vector<OUT_DATATYPE> FFN_MatMul1_Out_cpu(M_TGT * FFN_HIDDEN);
        cpu_matmul(FFN_In_16bit_cpu, FFN_W1, FFN_MatMul1_Out_cpu, M_TGT, K_MODEL, FFN_HIDDEN);

        std::vector<IN_DATATYPE> FFN_ReLU_Out_16bit_cpu(M_TGT * FFN_HIDDEN);
        for(int i = 0; i < M_TGT; ++i) {
            for(int j = 0; j < FFN_HIDDEN; ++j) {
                int index = i * FFN_HIDDEN + j;
                ACC_DATATYPE biased_val_64 = (ACC_DATATYPE)FFN_MatMul1_Out_cpu[index] + FFN_B1[j];
                OUT_DATATYPE relu_32bit = std::max((OUT_DATATYPE)0, (OUT_DATATYPE)biased_val_64);
                FFN_ReLU_Out_16bit_cpu[index] = static_cast<IN_DATATYPE>(std::clamp(relu_32bit, (OUT_DATATYPE)INT16_MIN, (OUT_DATATYPE)INT16_MAX));
            }
        }
        
        std::vector<OUT_DATATYPE> FFN_Out_cpu(M_TGT * K_MODEL);
        cpu_matmul(FFN_ReLU_Out_16bit_cpu, FFN_W2, FFN_Out_cpu, M_TGT, FFN_HIDDEN, K_MODEL);

        for(int i = 0; i < M_TGT; ++i) {
            for(int j = 0; j < K_MODEL; ++j) {
                FFN_Out_cpu[i * K_MODEL + j] += FFN_B2[j];
            }
        }

        // === CPU STEP 6: Add & Norm 3 ===
        for(int i=0; i<M_TGT*K_MODEL; ++i) { Final_CPU_Out[i] = FFN_Out_cpu[i] + AddNorm2_Out_cpu[i]; }
        cpu_layer_norm(Final_CPU_Out, M_TGT, K_MODEL);
        
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        float power_stop = get_power_in_watts();
        if (iter >= n_warmup_iterations) {
            cpu_time_total += std::chrono::duration_cast<std::chrono::microseconds>(cpu_stop - cpu_start).count();
            if (power_start > 0 && power_stop > 0) {
                cpu_power_total += (power_start + power_stop) / 2.0;
            }
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
        for(int i=0; i< M_TGT * K_MODEL; ++i) {
            if(abs(Final_NPU_Out[i] - Final_CPU_Out[i]) > 5) { 
                if(errors < 10) { 
                    std::cout << "ERROR: index=" << i << ", npu=" << Final_NPU_Out[i]
                              << ", ref=" << Final_CPU_Out[i] 
                              << ", diff=" << (Final_NPU_Out[i] - Final_CPU_Out[i]) << std::endl;
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
