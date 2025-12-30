#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

void run_1_chunk_fp32_syrk(
    cublasHandle_t h, cublasFillMode_t uplo,
    int N, int K,
    const float* dX, float* dC,
    float alpha, float beta);

void run_1_chunk_fp32_gemm(
    cublasHandle_t h,
    int N, int K,
    const float* dX,
    float* dC,
    float alpha, float beta);

void run_1_chunk_gemm_ex(
    cublasHandle_t h,
    int N, int K,
    const void* dX, cudaDataType Atype,
    float* dC,
    float alpha, float beta,
    cublasComputeType_t computeType);

template <typename TO>
__global__ void cast_f32_to(const float *in, TO *out, size_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx < n) out[idx] = (TO)in[idx];
}
