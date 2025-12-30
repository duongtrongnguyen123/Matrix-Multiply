#include "compute/xtx_cublasLt.h"
#include "compute/utils.h"

#include <cublasLt.h>
#include <vector>
#include <stdexcept>
#include <algorithm>


void run_1_chunk_fp32_xtx_cublaslt(
    int N, int K,
    const float* dX_rowmajor,  // [K x N] row-major contiguous
    float* dC_rowmajor,        // [N x N] row-major contiguous
    float alpha, float beta,
    cudaStream_t stream,
    void* workspace,
    size_t workspace_bytes,
    bool /*enable_tf32*/,      // ignored on older headers
    int heuristic_trials
) {
    static cublasLtHandle_t lt = nullptr;
    static bool inited = false;
    if (!inited) {
        cublaslt_check(cublasLtCreate(&lt), "cublasLtCreate");
        inited = true;
    }

    // Reinterpret row-major X(KxN) as column-major X_cm(NxK) (same bytes).
    // Want: C = X^T X (row-major) == X_cm * X_cm^T (column-major).
    const int64_t m = N;   // rows of C_cm
    const int64_t n = N;   // cols of C_cm
    const int64_t k = K;   // reduction dim

    // A = X_cm (m x k), opA = N
    // B = X_cm (m x k), opB = T  => (k x n)
    cublasOperation_t opA = CUBLAS_OP_N;
    cublasOperation_t opB = CUBLAS_OP_T;

    const int64_t ldX = m; // X_cm is (m x k) col-major => ld = m
    const int64_t ldC = m; // C_cm is (m x n) col-major => ld = m

    cublasLtMatmulDesc_t matmul = nullptr;
    cublasLtMatrixLayout_t xLayout = nullptr;
    cublasLtMatrixLayout_t cLayout = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    cublaslt_check(cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                   "cublasLtMatmulDescCreate");
    cublaslt_check(cublasLtMatmulDescSetAttribute(matmul, CUBLASLT_MATMUL_DESC_TRANSA,
                                                  &opA, sizeof(opA)),
                   "set TRANSA");
    cublaslt_check(cublasLtMatmulDescSetAttribute(matmul, CUBLASLT_MATMUL_DESC_TRANSB,
                                                  &opB, sizeof(opB)),
                   "set TRANSB");

    cublaslt_check(cublasLtMatrixLayoutCreate(&xLayout, CUDA_R_32F, m, k, ldX),
                   "cublasLtMatrixLayoutCreate X");
    cublaslt_check(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32F, m, n, ldC),
                   "cublasLtMatrixLayoutCreate C");

    cublaslt_check(cublasLtMatmulPreferenceCreate(&pref),
                   "cublasLtMatmulPreferenceCreate");
    cublaslt_check(cublasLtMatmulPreferenceSetAttribute(
                       pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                       &workspace_bytes, sizeof(workspace_bytes)),
                   "set MAX_WORKSPACE");

    bool own_ws = false;
    void* ws = workspace;
    if (!ws && workspace_bytes > 0) {
        cuda_check(cudaMalloc(&ws, workspace_bytes), "cudaMalloc cublasLt workspace");
        own_ws = true;
    }

    int trials = std::max(1, heuristic_trials);
    std::vector<cublasLtMatmulHeuristicResult_t> results(trials);
    int returned = 0;

    cublaslt_check(cublasLtMatmulAlgoGetHeuristic(
                       lt, matmul,
                       xLayout, xLayout,
                       cLayout, cLayout,
                       pref, trials,
                       results.data(), &returned),
                   "cublasLtMatmulAlgoGetHeuristic");

    if (returned <= 0) {
        if (own_ws) cuda_check(cudaFree(ws), "cudaFree workspace");
        throw std::runtime_error("[cublasLt] no heuristic algo returned");
    }

    int best = -1;
    for (int i = 0; i < returned; ++i) {
        if (results[i].state == CUBLAS_STATUS_SUCCESS) { best = i; break; }
    }
    if (best < 0) {
        if (own_ws) cuda_check(cudaFree(ws), "cudaFree workspace");
        throw std::runtime_error("[cublasLt] all heuristic algos invalid");
    }

    cublaslt_check(cublasLtMatmul(
                       lt,
                       matmul,
                       &alpha,
                       dX_rowmajor, xLayout,    // A = X_cm (reinterpret)
                       dX_rowmajor, xLayout,    // B = X_cm
                       &beta,
                       dC_rowmajor, cLayout,    // C in
                       dC_rowmajor, cLayout,    // D out
                       &results[best].algo,
                       ws, workspace_bytes,
                       stream),
                   "cublasLtMatmul XTX");

    if (own_ws) cuda_check(cudaFree(ws), "cudaFree workspace");
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (xLayout) cublasLtMatrixLayoutDestroy(xLayout);
    if (cLayout) cublasLtMatrixLayoutDestroy(cLayout);
    if (matmul) cublasLtMatmulDescDestroy(matmul);
}
