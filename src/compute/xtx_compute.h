#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <config/xtx_config.h>
#include <generate/xtx_generate.h>


struct ComputeParams {
    const ChunkingCfg& chunking;
    const std::vector<DeviceCfg>& devices;
    const ModeCfg& mode;
    const ComputeScalars& scalars;
    int64_t N;  // from matrix config
    bool double_buffering = true;
};

// Pre-allocated GPU buffers to avoid malloc/free overhead in hot path
struct GpuBuffers {
    int device_id = -1;

    // Output matrix C (N x N)
    float* dC = nullptr;
    size_t bytes_C = 0;

    // Input buffers (for double buffering: ping/pong, for single: just one)
    float* dX_ping = nullptr;
    float* dX_pong = nullptr;  // nullptr if single buffering

    // Casted buffers for fp16
    __half* dXh_ping = nullptr;
    __half* dXh_pong = nullptr;

    // Casted buffers for bf16
    __nv_bfloat16* dXb_ping = nullptr;
    __nv_bfloat16* dXb_pong = nullptr;

    size_t max_chunk_elems = 0;
    bool is_double_buffering = false;

    // Allocate all buffers based on config
    void allocate(int dev_id, int64_t N, int64_t rows_per_chunk,
                  const std::string& dtype, bool double_buffering);

    // Free all buffers
    void free();

    // Reset dC to zero (call before each compute iteration)
    void reset_output(cudaStream_t stream);
};

struct GpuTimeTotal {
    float gemm_ms = 0.f;
    float h2d_ms  = 0.f;
    float cast_ms = 0.f;
    // Actual elapsed time (accounts for overlap)
    float total_elapsed_ms = 0.f;
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
    float* C_out_row_major,
    std::vector<GpuBuffers>& gpu_buffers
);




