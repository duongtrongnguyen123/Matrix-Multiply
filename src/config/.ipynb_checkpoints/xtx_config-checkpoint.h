#pragma once 
#include <cstdint>
#include <string>
#include <vector>

struct MatrixCfg {
    int64_t M;
    int64_t N;
    uint64_t seed;
    std::string dtype_storage;
    std::string layout;
};

struct BenchmarkCfg {
    int repeats = 1;
    int warmup_iters = 0;
};

struct ChunkingCfg {
    int64_t rows_per_chunk;
    int pinned_buffers;
    double pinned_buffer_gb;
};

struct NodeFrac {
    int node = -1;
    double frac = 0.0;
};

struct HostMemoryCfg {
    std::string numa_mode = "manual"; // "manual" | "auto" (auto thì tuỳ mày xử lý sau)
    std::vector<NodeFrac> placement;  // must sum to 1

    int threads_per_node = 8;
    int max_threads = 0;              // 0 = no cap
    bool pin_threads = true;
    bool numa_aware = true;
};

struct DeviceCfg {
    int device_id;
    std::string backend;   // "cublas"
    bool use_streams;
    int streams;
};

struct ModeCfg {
    std::string name;
    std::string input_dtype;
    std::string algorithm = "syrk";
    std::string triangle;
    std::string compute;
    std::string cublas_math_mode;
    std::string accumulate;
    bool cast_on_gpu = true;
};

struct ComputeScalars {
    float alpha = 1.0f;
    float beta_first = 0.0f;
    float beta_rest  = 1.0f;
};

struct Config {
    std::string name;

    MatrixCfg matrix;
    ChunkingCfg chunking;
    BenchmarkCfg benchmark;

    HostMemoryCfg host_memory;    //numa nodes
    std::vector<DeviceCfg> devices;
    std::vector<ModeCfg> modes; // compute-only
    ComputeScalars compute_scalars; 
};

Config load_config_yaml(const std::string& path);


