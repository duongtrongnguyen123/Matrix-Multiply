#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <config/xtx_config.h>

struct ComputeParams {
    int64_t N;
    int64_t rows_per_chunk;
    int device_id;
    std::string input_dtype;
    std::string compute;
    std::string accumulate;
    std::string cublas_math_mode;
    std::string algorithm;
    std::string triangle;
    float alpha;
    float beta_first;
    float beta_rest;
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
void compute_xtx_gpu_mode(
    const ComputeParams& params,
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



