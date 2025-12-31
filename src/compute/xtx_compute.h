#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <config/xtx_config.h>
#include <generate/xtx_generate.h>


struct ComputeParams {
    int64_t N;
    int64_t rows_per_chunk;
    std::vector<DeviceCfg> devices;
    std::string name;
    std::string input_dtype;
    std::string algorithm;
    std::string triangle;
    std::string compute;
    std::string cublas_math_mode;
    std::string accumulate;
    
    float alpha;
    float beta_first;
    float beta_rest;
};

struct GpuTimeTotal {
    float gemm_ms = 0.f;
    float h2d_ms  = 0.f;
    float cast_ms = 0.f;
};

/*
 * Compute C = X^T X on GPU.
 *
 * Inputs:
 *   X_local   : host pointer, row-major [rows_local x N]
 *   X_remote  : host pointer, row-major [rows_remote x N]
 *
 * Output:
 *   C_out_rowmajor : host pointer, row-major [N x N]
 *
 * Notes:
 * - Uses chunked H2D copies (cfg.rows_per_chunk)
 * - Supports SYRK (half compute) or GEMM
 * - Accumulates in FP32
 */

std::vector<GpuTimeTotal> compute_xtx_multi_device(
    const ComputeParams& params,
    const GeneratedMatrix& X,
    float* C_out_row_major
);




