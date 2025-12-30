#include "compute/xtx_cublas.h"
#include "compute/utils.h"   

void run_1_chunk_fp32_syrk(
    cublasHandle_t h, cublasFillMode_t uplo,
    int N, int K,
    const float* dX, float* dC,
    float alpha, float beta
) {
    cublas_check(
        cublasSsyrk(h, uplo, CUBLAS_OP_N,
                    N, K,
                    &alpha,
                    dX, N,
                    &beta,
                    dC, N),
        "cublasSsyrk"
    );
}

void run_1_chunk_fp32_gemm(
    cublasHandle_t h,
    int N, int K,
    const float* dX,
    float* dC,
    float alpha, float beta
) {
    cublas_check(
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                    N, N, K,
                    &alpha,
                    dX, N,
                    dX, N,
                    &beta,
                    dC, N),
        "cublasSgemm"
    );
}

void run_1_chunk_gemm_ex(
    cublasHandle_t h,
    int N, int K,
    const void* dX, cudaDataType Atype,
    float* dC,
    float alpha, float beta,
    cublasComputeType_t computeType
) {
    cublas_check(
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T,
                     N, N, K,
                     &alpha,
                     dX, Atype, N,
                     dX, Atype, N,
                     &beta,
                     dC, CUDA_R_32F, N,
                     computeType,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx"
    );
}
