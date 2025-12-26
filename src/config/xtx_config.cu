#include <yaml-cpp/yaml.h>
#include "config/xtx_config.h"

#include <cstdint>
#include <string>

Config load_config_yaml(const std::string& path) { 
    YAML::Node root = YAML::LoadFile(path);
    Config cfg;
    
    MatrixCfg matrix{};
    ChunkingCfg chunking{};
    BenchmarkCfg benchmark{};

    HostMemoryCfg host_memory{};
    std::vector<DeviceCfg> devices;
    std::vector<ModeCfg> modes; // compute-only
    ComputeScalars compute_scalars{}; 

    if (root["experiment"]) {
        auto e = root["experiment"];
        //if (e["name"]) cfg.name = e["name"].as<std::string>();
        if (e["seed"]) matrix.seed = e["seed"].as<uint64_t>();
        if (e["repeats"])      benchmark.repeats      = e["repeats"].as<int>();
        if (e["warmup_iters"]) benchmark.warmup_iters = e["warmup_iters"].as<int>();
    }

    if (root["matrix"]) {
        auto m = root["matrix"];
        if (m["M"]) matrix.M = m["M"].as<int64_t>();
        if (m["N"]) matrix.N = m["N"].as<int64_t>();
        if (m["layout"]) matrix.layout = m["layout"].as<std::string>();
    }
    

    if (root["chunking"]) {
        auto c = root["chunking"];
        if (c["rows_per_chunk"])   chunking.rows_per_chunk   = (c["rows_per_chunk"]).as<int64_t>();
        if (c["pinned_buffers"])   chunking.pinned_buffers   = c["pinned_buffers"].as<int>();
        if (c["pinned_buffer_gb"]) chunking.pinned_buffer_gb = c["pinned_buffer_gb"].as<double>();
    }

    if (root["host_memory"]) {
        auto hm = root["host_memory"];

        if (hm["numa_mode"]) host_memory.numa_mode = to_lower(hm["numa_mode"].as<std::string>());

        if (hm["threads_per_node"]) host_memory.threads_per_node = hm["threads_per_node"].as<int>();
        if (hm["max_threads"])      host_memory.max_threads      = hm["max_threads"].as<int>();
        if (hm["pin_threads"])      host_memory.pin_threads      = hm["pin_threads"].as<bool>();
        if (hm["numa_aware"])       host_memory.numa_aware       = hm["numa_aware"].as<bool>();

        if (hm["placement"]) {
            auto pl = hm["placement"];

            host_memory.placement.clear();
            host_memory.placement.reserve(pl.size());

            for (const auto& it:pl) {
                NodeFrac nf;

                nf.node = it["node"].as<int>();
                nf.frac = it["frac"].as<double>();

                require_cfg(nf.node >= 0, "host_memory.placement[" + std::to_string(i) + "].node must be >= 0");
                require_cfg(nf.frac > 0.0, "host_memory.placement[" + std::to_string(i) + "].frac must be > 0");

                host_memory.placement.push_back(nf);
            }

            // validate sum(frac) ~= 1
            double sum = 0.0;
            for (const auto& nf : host_memory.placement) sum += nf.frac;

            const double eps = 1e-6;
            require_cfg(std::fabs(sum - 1.0) <= eps,
                        "`host_memory.placement` fractions must sum to 1 (got " + std::to_string(sum) + ")");
        }
    }
    

    if (root["devices"]) {
        auto dlist = root["devices"];

        for (const auto& d:dlist) {
            DeviceCfg dev{};

            if (d["device_id"]) dev.device_id = d["device_id"].as<int>();
            if (d["backend"])   dev.backend = to_lower(d["backend"].as<std::string>());

            if (d["use_streams"]) dev.use_streams = d["use_streams"].as<bool>();
            if (d["streams"])     dev.streams = d["streams"].as<int>();


            devices.push_back(dev);
        }
    }

    for (const auto& m: root["modes"]) {
        ModeCfg mode;

        mode.name             = m["name"].as<std::string>();
        mode.input_dtype      = m["input_dtype"].as<std::string>();
        mode.compute          = m["compute"].as<std::string>();
        mode.accumulate       = m["accumulate"].as<std::string>();
        mode.cublas_math_mode = m["cublas_math_mode"].as<std::string>();
        mode.algorithm        = m["algorithm"].as<std::string>();
        //mode.cast_on_gpu      = m["cast_on_gpu"].as<std::bool>();
        cfg.modes.push_back(std::move(mode));
    }
    
    if (root["xtx"]) {
        auto x = root["xtx"];
        benchmark.alpha      = x["alpha"].as<float>();
        benchmark.beta_first = x["beta_first"].as<float>();
        benchmark.beta_rest  = x["beta_rest"].as<float>();

    }
    if (matrix.M <= 0 || matrix.N <= 0) throw std::runtime_error("Invalid matrix size!");
    if (matix.layout != "row_major") {
        throw std::runtime_error("Only support row-major");
    }

    cfg.matrix = matrix;
    cfg.chunking = chunking;
    cfg.benchmark = benchmark;

    cfg.devices = std::move(devices);
    cfg.modes = std::move(modes);

    cfg.compute_scalars = compute_scalars;
    return cfg;
}



