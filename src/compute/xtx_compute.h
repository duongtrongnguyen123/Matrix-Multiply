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
static void compute_xtx_gpu_mode(
    const ComputeParams& params, int device_id,
    const float *X_local, std::int64_t rows_local,
    const float *X_remote, std::int64_t rows_remote,
    float *C_out_row_major, bool copy_back
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

void compute_xtx_multi_gpu(
    const ComputeParams& params,
    const GeneratedMatrix& X,
    float* C_out_row_major
);




