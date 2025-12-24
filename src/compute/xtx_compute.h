#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Config;

struct ModeCfg;

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
void compute_xtx_gpu_mode(
    const Config& cfg,
    const ModeCfg& mode,
    const float* X_local,
    std::int64_t rows_local,
    const float* X_remote,
    std::int64_t rows_remote,
    float* C_out_rowmajor
);

/*
 * Convenience wrapper.
 * Uses cfg.modes[0] if present, otherwise defaults to FP32.
 */
void compute_xtx_gpu(
    const Config& cfg,
    const float* X_local,
    std::int64_t rows_local,
    const float* X_remote,
    std::int64_t rows_remote,
    float* C_out_rowmajor
);

/*
 * Save a row-major float32 matrix (N x N) as NumPy .npy
 *
 * Returns true on success.
 */
bool save_npy_f32(
    const std::string& path,
    const float* data_rowmajor,
    std::int64_t N
);

