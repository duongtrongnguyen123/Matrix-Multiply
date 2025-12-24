#include <xtx_config.h>
#include <xtx_generate.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudafp16.h>
#include <cudabf16.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("[cuda] ") + msg + ": " + cudaGetErrorString(e));
    }
}

static inline void cublas_check(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("[cublas] ") + msg + " (status=" + std::to_string((int)s) + ")");
    }
}

static inline bool str_eq(const std::string& a, const char* b) {
    return a == std::string(b);
}

static inline cublasMath_t parse_math_mode(const std::string& s) {
    if (s.empty()) return CUBLAS_DEFAULT_MATH;

    if (s == "default") return CUBLAS_DEFAULT_MATH;
    if (s == "tf32") return CUBLAS_TF32_TENSOR_OP_MATH;
    if (s == "pedantic") return CUBLAS_PEDANTIC_MATH;
#if define(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) 
    if (t == "disallow_reduced_precision_reduction") return CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
#endif 
    return CUBLAS_DEFAULT_MATH;
}

static inline cublasFillMode_t parse_triangle(const std::string& tri) {
    if (tri == "upper") return CUBLAS_FILL_MODE_UPPER;
    return CUBLAS_FILL_MODE_LOWER;
}

__global__ void mirror_triangle(float* C, int N, int upper_from_lower) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    if (i == j) return;

    if (upper_from_lower) {
        if (i < j) C[i + j * n] = C[j + i * n]; // C[i, j] = C[j, i]
    }
    else {
        if (i > j) C[j + i * n] = C[i + j * n]; // C[j, i] = C[i, j]
    }
}

static void run_1_chunk_fp32_syrk(
        cublasHandle_t h, cublasFillMode_t uplo,
        int N, int K, 
        const float* dX, float* dC,
        float alpha, float beta
        ) {
    cublas_check(
            cublasSsyrk(h, uplo, CUBLAS_OP_N,
                        N, K,        // size of C, inner dimension
                        &alpha,
                        dX, N,       // pointer to X, leading dim A
                        &beta, 
                        dC, N),      // pointer to C, leading dim C
            "cublasSsyrk"
            );
}

static void run_1_chunk_fp32_gemm(
        cublasHandle_t h,
        int N, int K,
        const float* dX, cudaDataType Atype, 
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

template <typename TO>
__global__ void cast_f32_to(const float *in, TO *out, size_t n) {
    int idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (TO) in[idx];
}

static void run_1_chunk_gemm_ex(
        cublasHandle_t h, 
        int N, int K,
        const float *dX, cudaDataType Atype,
        float *dC, 
        float alpha, float beta,
        cublasComputeType_t compute
        ) {
    cublas_check(
            cublasSgemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T,
                          N, N, K,
                          &alpha, 
                          dX, Atype, N,
                          dX, Atype, N,
                          &beta,
                          dC, CUDA_R_32F, N,
                          compute,
                          CUBLAS_DEFAULT_TENSOR_OP
                          ),
            "cublasSgemmEx"
            );
}

void compute_xtx_gpu_mode(
        const Config& cfg,
        const ModeCfg& mode,
        const float *Xlocal, std::int64_t rows_local,
        const float *Xremote, std::int64_t rows_remote,
        float *C_out_row_major
        ) {
    const int m_total = rows_local + rows_remote;
    const int N = (int) cfg.N;

    if (!C_out_row_major) return std::runtime_error("C_out_row_major must not be null");

    cuda_check(cudaSetDevice(cfg.device_id), "CudaSetDevice");
    
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetStream(handle, stream), "cublasSetStream");

    cublasMath_t math_mode = parse_math_mode(mode.cublas_math_mode);
    cublas_check(cublasSetMathMode(handle, math_mode), "cublasSetMathMode");

    // output C on device
    float* dC = nullptr;
    size_t bytesC = (size_t) N * N * sizeof(float);
    cuda_check(cudaMalloc((void**)&dC, bytesC), "cudaMalloc dC");
    cuda_check(cudaMemsetAsync(dC, 0, bytesC, stream), "cudaMemsetAsync dC");

    const int64_t rows_per_chunk = cfg.rows_per_chunk;
    const size_t max_chunk_elems = (size_t) rows_per_chunk * N;

    float *dXf = nullptr;
    cuda_check(cudaMalloc((void**)&dXf), max_chunk_elems * sizeof(float), "cudaMalloc");

    __half *dXh = nullptr;
    __nv_bfloat16 *dXb = nullptr;

    std::string dtype = cfg.input_dtype;

    const bool want_fp16 = (dtype == "fp16");
    const bool want_bf16 = (dtype == "bf16");
    const bool want_fp32 = (!want_fp16 && !want_bf16);

    if (want_fp16) cuda_check(cudaMalloc((void**)&dXh), max_chunk_elems * sizeof(__half), "cudaMallocfp16");
    if (want_bf16) cuda_check(cudaMalloc((void**)&dXb), max_chunk_elems * sizeof(__nv_bfloat16), "cudaMallocbf16");

    const cublasFillMode_t uplo = parse_triangle(cfg.triangle);

    // X points to DRAM, diff with dXf
    auto process_source = [&](const float* X, int64_t rows) {
        int64_t done = 0;
        while (done < rows) {
            int64_t K = std::min<int64_t> (rows_per_chunk, rows-done);
            const size_t elems = (size_t) K * N;

            // H2D copy
            cuda_check(
                    cudaMemcpyAsync(dXf, X + (size_t) done * N, 
                                    elems * sizeof(float), 
                                    cudaMemcpyHostToDevice, stream
                                    ), 
                    "cudaMemcpuAsync dXf -> X"
            );

            const float alpha = cfg.alpha;
            const float beta  = (done == 0 && X == X_local) ? cfg.beta_first : cfg.beta_rest;

            if (want_fp32) {
                if (cfg.algorithm == "syrk") {
                    run_1_chunk_fp32_syrk(handle, uplo,
                                          N, (int) K, 
                                          dXf, dC,
                                          alpha, beta);
                }
                else run_1_chunk_fp32_gemm(handle, 
                                           N, (int) K,
                                           dXf, dC,
                                           alpha, beta); 
            }
            else if (want_fp16) {
                int thread = 256;
                int blocks = (int)((elems + thread - 1) / thread);
                cast_f32_to<<<blocks, threads, 0, stream>>>(dXf, dXh, elems);
                cuda_check(cudaGetLastError(), "cast fp32 to fp16 kernel");
                run_1_chunk_gemm_ex(handle, 
                                    N, (int) K,
                                    dXh, CUDA_R_16F, 
                                    dC, 
                                    alpha, beta, 
                                    CUBLAS_COMPUTE_32F);
            }
            else {
                int thread = 256;
                int blocks = (int)((elems + thread - 1) / thread);
                cast_f32_to<<<blocks, threads, 0, stream>>>(dXf, dXb, elems);
                cuda_check(cudaGetLastError(), "cast fp32 to bf16 kernel");
                run_1_chunk_gemm_ex(handle,
                                    N, (int) K,
                                    dXb, CUDA_R_16BF,
                                    dC, 
                                    alpha, beta,
                                    CUBLAS_COMPUTE_32F);
            }
            
            done += K;
        }        
    }

    if (rows_local > 0)  process_source(X_local, rows_local);
    if (rows_remote > 0) process_source(X_remote, rows_remote);
    
    if (want_fp32 && cfg.algorithm == "syrk") {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        const int upper_from_lower = (uplo == CUBLAS_FILL_MODE_LOWER) ? 1 : 0;
        mirror_triangle<<<grid, block, 0, stream>>>(dC, N, upper_from_lower);
        cuda_check(cudaGetLastError(), "mirror_triangle kernel");
    }

    // storage to host from device
    // interpreting col-major as row-major
    cuda_check(cudaMemcpyAsync(C_out_row_major, dC, bytesC, cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    if (dXb) cudaFree(dXb);
    if (dXh) cudaFree(dXh);
    cudaFree(dXf);
    cudaFree(dC);
    cublasDestroy(handle);
    cublasStreamDestroy(stream);
}

void compute_xtx_gpu(
        const Config& cfg,
        const float* X_local, std::int64_t rows_local,
        const float* X_remote, std::int64_t rows_remote,
        float* C_out_row_major
        ) {
    ModeCfg m;
    if (!cfg.modes.empty()) {
        m = cfg.modes[0];
    }
    else {
        m.name = "default_fp32";
        m.input_dtype = "fp32";
        m.cublas_math_mode = "default";
        m.accumulate = "fp32";
        m.cast_on_gpu = true;
    }
    compute_xtx_gpu_mode(cfg, m, X_local, rows_local, X_remote, rows_remote, C_out_row_major)
}

bool save_npy_fp32(const std::string& path, const float* data_row_major, std::int64_t N) {
    if (!data_rowmajor || N <= 0) return false;

    // .npy v1.0 header (little-endian float32, C-order)
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         std::to_string(N) + ", " + std::to_string(N) + "), }";

    // Pad to 16-byte alignment with spaces and newline at end
    size_t header_len = header.size() + 1; // + newline
    size_t preamble = 10; // magic(6) + ver(2) + headerlen(2)
    size_t pad = 16 - ((preamble + header_len) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');

    uint16_t hlen = (uint16_t)header.size();

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    f.write((const char*)magic, 6);
    unsigned char ver[2] = {1, 0};
    f.write((const char*)ver, 2);
    f.write((const char*)&hlen, 2);
    f.write(header.data(), (std::streamsize)header.size());

    const size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    f.write((const char*)data_rowmajor, (std::streamsize)bytes);

    return (bool)f;

}


