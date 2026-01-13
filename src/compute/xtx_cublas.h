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

// FP64 compute for ground truth reference
void run_1_chunk_fp64_gemm(
    cublasHandle_t h,
    int N, int K,
    const double* dX,
    double* dC,
    double alpha, double beta);

template <typename TO>
__global__ void cast_f32_to(const float *in, TO *out, size_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx < n) out[idx] = (TO)in[idx];
}

// Cast fp64 output back to fp32
__global__ inline void cast_f64_to_f32(const double *in, float *out, size_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (idx < n) out[idx] = (float)in[idx];
}
