#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// X is row-major contiguous: shape (K x N)  (K = rows_chunk)
// C is row-major contiguous: shape (N x N)
//
// Computes: C += alpha * (X^T * X) + beta * C   (same semantics as your GEMM path)
//
// workspace can be nullptr (then it will cudaMalloc/cudaFree internally).
void run_1_chunk_fp32_xtx_cublaslt(
    int N, int K,
    const float* dX_rowmajor,  // [K x N]
    float* dC_rowmajor,        // [N x N]
    float alpha, float beta,
    cudaStream_t stream,
    void* workspace = nullptr,
    size_t workspace_bytes = 256ull * 1024 * 1024,
    bool enable_tf32 = true,
    int heuristic_trials = 8
);
