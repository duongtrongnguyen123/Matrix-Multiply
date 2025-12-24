#pragma once
#include <yaml-cpp/yaml.h>
#include "configs/xtx_config.h"

#include <cstdint>
#include <string>

Config load_config_yaml(const std::string& path) { 
    YAML::Node root = YAML::LoadFile(path);
    Config cfg;

    if (root["experiment"]) {
        auto e = root["experiment"];
        if (e["name"]) cfg.name = e["name"].as<std::string>();
        if (e["seed"]) cfg.seed = e["seed"].as<uint64_t>();
        if (e["repeats"])      cfg.repeats      = e["repeats"].as<int>();
        if (e["warmup_iters"]) cfg.warmup_iters = e["warmup_iters"].as<int>();
    }

    if (root["matrix"]) {
        auto m = root["matrix"];
        if (m["M"]) cfg.M = m["M"].as<int64_t>();
        if (m["N"]) cfg.N = m["N"].as<int64_t>();
        if (m["layout"]) cfg.layout = m["layout"].as<std::string>();
    }
    
    if (root["numa"]) {
        auto n = root["numa"];
        if (n["gpu_local_node"]) cfg.gpu_local_node = n["gpu_local_node"].as<int>();
        if (n["remote_node"])    cfg.remote_node    = n["remote_node"].as<int>();
        if (n["split_ratio"])    cfg.split_ratio    = n["split_ratio"].as<double>();
    } else {
    // If remote_node not given, infer it as 1 - gpu_local_node later.
    }

    if (root["chunking"]) {
        auto c = root["chunking"];
        if (c["rows_per_chunk"])   cfg.rows_per_chunk   = static_cast<int64_t>(c["rows_per_chunk"]);
        if (c["pinned_buffers"])   cfg.pinned_buffers   = c["pinned_buffers"];
        if (c["pinned_buffer_gb"]) cfg.pinned_buffer_gb = c["pinned_buffer_gb"];
    }

    if (root["gpu"]) {
        auto g = root["gpu"];
        if (g["device_id"])   cfg.device_id   = g["device_id"];
        if (g["use_streams"]) cfg.use_streams = g["use_streams"];
        if (g["streams"])     cfg.stream      = g["streams"];
    }

    if (root["xtx"]) {
        auto x = root["xtx"];
        if (x["algorithm"])  cfg.algorithm  = x["algorithm"].as<std::string>();
        if (x["triangle"])   cfg.triangle   = x["triangle"].as<std::string>();
        if (x["alpha"])      cfg.alpha      = x["alpha"].as<float>();
        if (x["beta_first"]) cfg.beta_first = x["beta_first"].as<float>();
        if (x["beta_rest"])  cfg.beta_rest  = x["beta_rest"].as<float>();
    }

    for (const auto& m: root["modes"]) {
        ModeCfg mode;

        mode.name             = m["name"].as<std::string>();
        mode.input_dtype      = m["input_dtype"].as<std::string>();
        mode.accumulate       = m["accumulate"].as<std::string>();
        mode.cublas_math_mode = m["cublas_math_mode"].as<std::string>();
        mode.cast_on_gpu      = m["cast_on_gpu"].as<std::string>();
        cfg.modes.push_back(std::move(mode));
    }
    
    if (cfg.remote_node == cfg.gpu_local_node) {
        cfg.remote_node = 1 - cfg.gpu_local_node;
    }

    if (cfg.M <= 0 || cfg.N <= 0) throw std::runtime_error("Invalid matrix size!");
    if (cfg.layout != "row_major") {
        throw std::runtime_error("Only support row-major");
    }
    return cfg;
}


