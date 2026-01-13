#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <config/xtx_config.h>
#include <generate/xtx_generate.h>


struct ComputeParams {
    const ChunkingCfg& chunking;
    const std::vector<DeviceCfg>& devices;
    const ModeCfg& mode;
    const ComputeScalars& scalars;
    int64_t N;  // from matrix config
    bool double_buffering = true;
    bool copy_back = false;  // copy result from GPU to host (D2H)
};

// Pre-allocated GPU buffers 
struct GpuBuffers {
    int device_id = -1;

    // Output matrix C (N x N)
    float* dC = nullptr;
    size_t bytes_C = 0;

    // Pinned host buffer for parallel D2H reduction
    float* h_C_partial = nullptr;

    // Input buffers (for double buffering: ping/pong, for single: just one)
    float* dX_ping = nullptr;
    float* dX_pong = nullptr;  // nullptr if single buffering

    // Casted buffers for fp16
    __half* dXh_ping = nullptr;
    __half* dXh_pong = nullptr;

    // Casted buffers for bf16
    __nv_bfloat16* dXb_ping = nullptr;
    __nv_bfloat16* dXb_pong = nullptr;

    // Casted buffers for fp64 (ground truth reference)
    double* dXd_ping = nullptr;
    double* dXd_pong = nullptr;
    double* dC_fp64 = nullptr;  // FP64 output buffer (will be cast to fp32 at end)

    size_t max_chunk_elems = 0;
    bool is_double_buffering = false;

    // ---- Pre-allocated CUDA resources ----
    // Streams
    cudaStream_t stream_h2d = nullptr;      // H2D transfer stream
    cudaStream_t stream_compute = nullptr;  // Compute stream (also used as single stream)

    // cuBLAS handle
    cublasHandle_t cublas_handle = nullptr;

    // Timing events (pre-allocated for max_chunks)
    int allocated_max_chunks = 0;
    std::vector<cudaEvent_t> gemm_start, gemm_stop;
    std::vector<cudaEvent_t> h2d_start, h2d_stop;
    std::vector<cudaEvent_t> cast_start, cast_stop;
    cudaEvent_t h2d_done = nullptr;
    cudaEvent_t compute_done = nullptr;
    cudaEvent_t overall_start = nullptr;
    cudaEvent_t overall_stop = nullptr;

    // Allocate all buffers based on config
    // M_total is needed to calculate max_chunks for event allocation
    void allocate(int dev_id, int64_t N, int64_t M_total, int64_t rows_per_chunk,
                  const std::string& dtype, bool double_buffering);

    // Free all buffers
    void free();

    // Reset dC to zero
    void reset_output(cudaStream_t stream);
};

struct GpuTimeTotal {
    float gemm_ms = 0.f;
    float h2d_ms  = 0.f;
    float cast_ms = 0.f;
    //float total_elapsed_ms = 0.f;
};

/*
 * Compute C = X^T X on GPU.
 *
 * Inputs:
 *   
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




