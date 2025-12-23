#pragma once 
#include <cstdint>
#include <string>
#include <vector>

struct ModeCfg {
    std::string name;
    std::string input_dtype;
    std::string cublas_math_mode;
    std::string accumulate;
    bool cast_on_gpu = true;
};

struct Config {
    std::string name;
    uint64_t seed = 1234;
    int repeats = 1;
    int warmup_iters = 0;

    int64_t M = 1500000;
    int64_t N = 10000;
    std::string layout = "row_major";

    int gpu_local_node = 1;
    int remote_node = 0;
    double split_ratio = 0.5;

    // chunking (compute-only)
    std::int64_t rows_per_chunk = 300000;
    int pinned_buffers = 1;
    double pinned_buffer_gb = 12.0;

    // gpu (compute-only)
    int device_id = 0;
    bool use_streams = true;
    int streams = 2;

    std::string algorithm = "syrk"; // syrk/gemm
    std::string triangle = "lower";
    float alpha = 1.0f;
    float beta_first = 0.0f;
    float beta_rest = 1.0f;

    // modes (compute-only)
    std::vector<ModeCfg> modes;    

};

Config load_config_yaml(const std::string& path);

