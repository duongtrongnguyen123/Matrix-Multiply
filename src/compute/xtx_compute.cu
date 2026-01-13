#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <thread>
#include <omp.h>


#include <compute/xtx_compute.h>
#include <compute/utils.h>
#include <compute/xtx_cublas.h>
#include <compute/xtx_cublasLt.h>


// ========== GpuBuffers implementation ==========

void GpuBuffers::allocate(int dev_id, int64_t N, int64_t M_total, int64_t rows_per_chunk,
                          const std::string& dtype, bool double_buffering) {
    device_id = dev_id;
    is_double_buffering = double_buffering;
    max_chunk_elems = static_cast<size_t>(rows_per_chunk) * static_cast<size_t>(N);

    cuda_check(cudaSetDevice(device_id), "cudaSetDevice in GpuBuffers::allocate");

    // Allocate output C (N x N)
    bytes_C = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dC), bytes_C), "cudaMalloc dC");

    // Allocate pinned host buffer for parallel D2H reduction
    cuda_check(cudaMallocHost(reinterpret_cast<void**>(&h_C_partial), bytes_C), "cudaMallocHost h_C_partial");

    // Allocate input buffers
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dX_ping), max_chunk_elems * sizeof(float)), "cudaMalloc dX_ping");

    if (double_buffering) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dX_pong), max_chunk_elems * sizeof(float)), "cudaMalloc dX_pong");
    }

    // Allocate casted buffers based on dtype
    const bool want_fp16 = (dtype == "fp16");
    const bool want_bf16 = (dtype == "bf16");

    if (want_fp16) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dXh_ping), max_chunk_elems * sizeof(__half)), "cudaMalloc dXh_ping");
        if (double_buffering) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&dXh_pong), max_chunk_elems * sizeof(__half)), "cudaMalloc dXh_pong");
        }
    }

    if (want_bf16) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dXb_ping), max_chunk_elems * sizeof(__nv_bfloat16)), "cudaMalloc dXb_ping");
        if (double_buffering) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&dXb_pong), max_chunk_elems * sizeof(__nv_bfloat16)), "cudaMalloc dXb_pong");
        }
    }

    const bool want_fp64 = (dtype == "fp64");
    if (want_fp64) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dXd_ping), max_chunk_elems * sizeof(double)), "cudaMalloc dXd_ping");
        if (double_buffering) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&dXd_pong), max_chunk_elems * sizeof(double)), "cudaMalloc dXd_pong");
        }
        // Allocate fp64 output buffer (N x N)
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&dC_fp64), static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(double)), "cudaMalloc dC_fp64");
    }

    // ---- Create CUDA streams ----
    cuda_check(cudaStreamCreateWithFlags(&stream_h2d, cudaStreamNonBlocking), "cudaStreamCreate stream_h2d");
    cuda_check(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking), "cudaStreamCreate stream_compute");

    // ---- Create cuBLAS handle ----
    cublas_check(cublasCreate(&cublas_handle), "cublasCreate");
    cublas_check(cublasSetStream(cublas_handle, stream_compute), "cublasSetStream");

    // ---- Create timing events ----
    // Calculate max_chunks based on M_total (worst case: all rows on this GPU)
    allocated_max_chunks = static_cast<int>((M_total + rows_per_chunk - 1) / rows_per_chunk);

    gemm_start.resize(allocated_max_chunks);
    gemm_stop.resize(allocated_max_chunks);
    h2d_start.resize(allocated_max_chunks);
    h2d_stop.resize(allocated_max_chunks);
    cast_start.resize(allocated_max_chunks);
    cast_stop.resize(allocated_max_chunks);

    for (int i = 0; i < allocated_max_chunks; ++i) {
        cuda_check(cudaEventCreate(&gemm_start[i]), "cudaEventCreate gemm_start");
        cuda_check(cudaEventCreate(&gemm_stop[i]), "cudaEventCreate gemm_stop");
        cuda_check(cudaEventCreate(&h2d_start[i]), "cudaEventCreate h2d_start");
        cuda_check(cudaEventCreate(&h2d_stop[i]), "cudaEventCreate h2d_stop");
        cuda_check(cudaEventCreate(&cast_start[i]), "cudaEventCreate cast_start");
        cuda_check(cudaEventCreate(&cast_stop[i]), "cudaEventCreate cast_stop");
    }

    // Sync events for double buffering
    cuda_check(cudaEventCreate(&h2d_done), "cudaEventCreate h2d_done");
    cuda_check(cudaEventCreate(&compute_done), "cudaEventCreate compute_done");
    cuda_check(cudaEventCreate(&overall_start), "cudaEventCreate overall_start");
    cuda_check(cudaEventCreate(&overall_stop), "cudaEventCreate overall_stop");
}

void GpuBuffers::free() {
    if (device_id < 0) return;

    cuda_check(cudaSetDevice(device_id), "cudaSetDevice in GpuBuffers::free");

    // Destroy timing events
    for (int i = 0; i < allocated_max_chunks; ++i) {
        if (gemm_start[i]) cudaEventDestroy(gemm_start[i]);
        if (gemm_stop[i]) cudaEventDestroy(gemm_stop[i]);
        if (h2d_start[i]) cudaEventDestroy(h2d_start[i]);
        if (h2d_stop[i]) cudaEventDestroy(h2d_stop[i]);
        if (cast_start[i]) cudaEventDestroy(cast_start[i]);
        if (cast_stop[i]) cudaEventDestroy(cast_stop[i]);
    }
    gemm_start.clear(); gemm_stop.clear();
    h2d_start.clear(); h2d_stop.clear();
    cast_start.clear(); cast_stop.clear();

    if (h2d_done) { cudaEventDestroy(h2d_done); h2d_done = nullptr; }
    if (compute_done) { cudaEventDestroy(compute_done); compute_done = nullptr; }
    if (overall_start) { cudaEventDestroy(overall_start); overall_start = nullptr; }
    if (overall_stop) { cudaEventDestroy(overall_stop); overall_stop = nullptr; }

    // Destroy cuBLAS handle
    if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; }

    // Destroy streams
    if (stream_h2d) { cudaStreamDestroy(stream_h2d); stream_h2d = nullptr; }
    if (stream_compute) { cudaStreamDestroy(stream_compute); stream_compute = nullptr; }

    // Free device memory
    if (dC_fp64)  { cudaFree(dC_fp64);  dC_fp64 = nullptr; }
    if (dXd_pong) { cudaFree(dXd_pong); dXd_pong = nullptr; }
    if (dXd_ping) { cudaFree(dXd_ping); dXd_ping = nullptr; }
    if (dXb_pong) { cudaFree(dXb_pong); dXb_pong = nullptr; }
    if (dXb_ping) { cudaFree(dXb_ping); dXb_ping = nullptr; }
    if (dXh_pong) { cudaFree(dXh_pong); dXh_pong = nullptr; }
    if (dXh_ping) { cudaFree(dXh_ping); dXh_ping = nullptr; }
    if (dX_pong)  { cudaFree(dX_pong);  dX_pong = nullptr; }
    if (dX_ping)  { cudaFree(dX_ping);  dX_ping = nullptr; }
    if (dC)       { cudaFree(dC);       dC = nullptr; }

    // Free pinned host buffer
    if (h_C_partial) { cudaFreeHost(h_C_partial); h_C_partial = nullptr; }

    device_id = -1;
    allocated_max_chunks = 0;
}

void GpuBuffers::reset_output(cudaStream_t stream) {
    if (dC && bytes_C > 0) {
        cuda_check(cudaMemsetAsync(dC, 0, bytes_C, stream), "cudaMemsetAsync dC reset");
    }
}

// ===== quick cublasLt XTX test knobs =====
static constexpr bool USE_CUBLASLT_XTX = false;
static constexpr bool CUBLASLT_ENABLE_TF32 = false;

// 256 MB workspace for cublasLt
static constexpr size_t CUBLASLT_WORKSPACE_BYTES = 256ull * 1024 * 1024;

__global__ void mirror_triangle(float* C, int N, int upper_from_lower) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    if (i == j) return;

    if (upper_from_lower) {
        if (i < j) C[i + j * N] = C[j + i * N]; // C[i, j] <- C[j, i]
    }
    else {
        if (i > j) C[j + i * N] = C[i + j * N]; // C[j, i] <- C[i, j]
    }
}

void compute_xtx_double_buffering (
        const ComputeParams& params, int device_id,
        const float *X_local, std::int64_t rows_local,
        const float *X_remote, std::int64_t rows_remote,
        float *C_out_row_major, bool copy_back,
        GpuBuffers& bufs,
        GpuTimeTotal& time
        ){
    const ModeCfg& mode = params.mode;
    const ComputeScalars& scalars = params.scalars;
    const int N = static_cast<int>(params.N);

    if (!C_out_row_major) {
        throw std::runtime_error("C_out_row_major must not be null");
    }


    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");

    // Use pre-allocated streams and handle
    cudaStream_t stream_h2d = bufs.stream_h2d;
    cudaStream_t stream_compute = bufs.stream_compute;
    cublasHandle_t handle = bufs.cublas_handle;

    // Set math mode (needs to be done each call as mode might change)
    cublasMath_t math_mode = parse_math_mode(mode.cublas_math_mode);
    cublas_check(cublasSetMathMode(handle, math_mode), "cublasSetMathMode");

    // Use pre-allocated buffers
    float* dC = bufs.dC;
    size_t bytes_C = bufs.bytes_C;
    bufs.reset_output(stream_compute);

    const int64_t rows_per_chunk = params.chunking.rows_per_chunk;

    // Double buffer for input: ping and pong (pre-allocated)
    float* dX_ping = bufs.dX_ping;
    float* dX_pong = bufs.dX_pong;

    __half* dXh_ping = bufs.dXh_ping;
    __half* dXh_pong = bufs.dXh_pong;
    __nv_bfloat16* dXb_ping = bufs.dXb_ping;
    __nv_bfloat16* dXb_pong = bufs.dXb_pong;

    // compute type
    const std::string& dtype = mode.name;

    const bool want_fp16 = (dtype == "fp16");
    const bool want_bf16 = (dtype == "bf16");
    const bool want_tf32 = (dtype == "tf32");
    const bool want_fp64 = (dtype == "fp64");
    const bool want_fp32 = (!want_fp16 && !want_bf16 && !want_tf32 && !want_fp64);

    // FP64 buffers
    double* dXd_ping = bufs.dXd_ping;
    double* dXd_pong = bufs.dXd_pong;
    double* dC_fp64 = bufs.dC_fp64;

    const cublasFillMode_t uplo = parse_triangle(mode.triangle);

    // Use pre-allocated events
    std::vector<cudaEvent_t>& gemm_start = bufs.gemm_start;
    std::vector<cudaEvent_t>& gemm_stop = bufs.gemm_stop;
    std::vector<cudaEvent_t>& h2d_start = bufs.h2d_start;
    std::vector<cudaEvent_t>& h2d_stop = bufs.h2d_stop;
    std::vector<cudaEvent_t>& cast_start = bufs.cast_start;
    std::vector<cudaEvent_t>& cast_stop = bufs.cast_stop;
    cudaEvent_t h2d_done = bufs.h2d_done;
    cudaEvent_t compute_done = bufs.compute_done;

    bool use_ping = true;

    // Build list of all chunks to process
    struct ChunkInfo {
        const float* src;
        int64_t offset;
        int64_t rows;
        bool is_first;
    };
    std::vector<ChunkInfo> chunks;

    auto add_chunks = [&](const float* X, int64_t total_rows, bool is_local) {
        int64_t done = 0;
        while (done < total_rows) {
            int64_t K = std::min<int64_t>(rows_per_chunk, total_rows - done);
            chunks.push_back({X, done, K, (done == 0 && is_local)});
            done += K;
        }
    };

    if (X_local && rows_local > 0) add_chunks(X_local, rows_local, true);
    if (X_remote && rows_remote > 0) add_chunks(X_remote, rows_remote, false);

    if (chunks.empty()) {
        // Nothing to process, return
        // All resources are pre-allocated and will be freed by GpuBuffers::free()
        return;
    }

    // Helper lambda to run compute on a buffer
    auto run_compute = [&](float* dXf, __half* dXh, __nv_bfloat16* dXb, double* dXd,
                           int64_t K, size_t elems, float beta, int cid) {
        const float alpha = scalars.alpha;

        if (want_fp64) {
            // Cast fp32 input to fp64
            int threads = 256;
            int blocks = static_cast<int>((elems + threads - 1) / threads);
            cudaEventRecord(cast_start[cid], stream_compute);
            cast_f32_to<<<blocks, threads, 0, stream_compute>>>(dXf, dXd, elems);
            cuda_check(cudaGetLastError(), "cast fp32 to fp64 kernel");
            cudaEventRecord(cast_stop[cid], stream_compute);

            // FP64 GEMM
            cudaEventRecord(gemm_start[cid], stream_compute);
            run_1_chunk_fp64_gemm(handle, N, static_cast<int>(K), dXd, dC_fp64,
                                  static_cast<double>(alpha), static_cast<double>(beta));
            cudaEventRecord(gemm_stop[cid], stream_compute);
        } else if (want_fp32) {
            cudaEventRecord(gemm_start[cid], stream_compute);
            if (mode.algorithm == "syrk") {   // FP32 syrk
                run_1_chunk_fp32_syrk(handle, uplo, N, static_cast<int>(K), dXf, dC, alpha, beta);
            } else {
                if (USE_CUBLASLT_XTX) {       // FP32 cublasLt
                    run_1_chunk_fp32_xtx_cublaslt(
                        N, static_cast<int>(K), dXf, dC, alpha, beta,
                        stream_compute, nullptr, CUBLASLT_WORKSPACE_BYTES, CUBLASLT_ENABLE_TF32);
                } else {                      // FP32 gemm
                    run_1_chunk_fp32_gemm(handle, N, static_cast<int>(K), dXf, dC, alpha, beta);
                }
            }
            cudaEventRecord(gemm_stop[cid], stream_compute);
        } else if (want_tf32) {
            cudaEventRecord(gemm_start[cid], stream_compute);
            if (mode.algorithm == "syrk") {   // TF32 syrk
                run_1_chunk_fp32_syrk(handle, uplo, N, static_cast<int>(K), dXf, dC, alpha, beta);
            } else {
                if (USE_CUBLASLT_XTX) {       // TF32 cublasLt
                    run_1_chunk_fp32_xtx_cublaslt(
                        N, static_cast<int>(K), dXf, dC, alpha, beta,
                        stream_compute, nullptr, CUBLASLT_WORKSPACE_BYTES, true);
                } else {                      // TF32 gemmex Tensor core
                    run_1_chunk_gemm_ex(handle, N, static_cast<int>(K), dXf, CUDA_R_32F, dC,
                                        alpha, beta, CUBLAS_COMPUTE_32F_FAST_TF32);
                }
            }
            cudaEventRecord(gemm_stop[cid], stream_compute);
        } else if (want_fp16) {               // FP16 tensor core
            int threads = 256;
            int blocks = static_cast<int>((elems + threads - 1) / threads);
            cudaEventRecord(cast_start[cid], stream_compute);
            cast_f32_to<<<blocks, threads, 0, stream_compute>>>(dXf, dXh, elems);
            cuda_check(cudaGetLastError(), "cast fp32 to fp16 kernel");
            cudaEventRecord(cast_stop[cid], stream_compute);

            cudaEventRecord(gemm_start[cid], stream_compute);
            run_1_chunk_gemm_ex(handle, N, static_cast<int>(K), dXh, CUDA_R_16F, dC,
                                alpha, beta, CUBLAS_COMPUTE_32F);
            cudaEventRecord(gemm_stop[cid], stream_compute);
        } else {                              // BF16 tensor core
            int threads = 256;
            int blocks = static_cast<int>((elems + threads - 1) / threads);
            cudaEventRecord(cast_start[cid], stream_compute);
            cast_f32_to<<<blocks, threads, 0, stream_compute>>>(dXf, dXb, elems);
            cuda_check(cudaGetLastError(), "cast fp32 to bf16 kernel");
            cudaEventRecord(cast_stop[cid], stream_compute);

            cudaEventRecord(gemm_start[cid], stream_compute);
            run_1_chunk_gemm_ex(handle, N, static_cast<int>(K), dXb, CUDA_R_16BF, dC,
                                alpha, beta, CUBLAS_COMPUTE_32F);
            cudaEventRecord(gemm_stop[cid], stream_compute);
        }
    };

    // Prime the pipeline: start H2D for first chunk
    {
        const auto& c = chunks[0];
        const size_t elems = static_cast<size_t>(c.rows) * static_cast<size_t>(N);
        cudaEventRecord(h2d_start[0], stream_h2d);
        cuda_check(cudaMemcpyAsync(dX_ping, c.src + c.offset * N, elems * sizeof(float),
                                   cudaMemcpyHostToDevice, stream_h2d), "H2D chunk 0");
        cudaEventRecord(h2d_stop[0], stream_h2d);
        cudaEventRecord(h2d_done, stream_h2d);
    }

    // Process all chunks with double buffering
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto& c = chunks[i];
        const size_t elems = static_cast<size_t>(c.rows) * static_cast<size_t>(N);
        const float beta = c.is_first ? scalars.beta_first : scalars.beta_rest;

        float* dXf_curr = use_ping ? dX_ping : dX_pong;
        float* dXf_next = use_ping ? dX_pong : dX_ping;
        __half* dXh_curr = use_ping ? dXh_ping : dXh_pong;
        __half* dXh_next = use_ping ? dXh_pong : dXh_ping;
        __nv_bfloat16* dXb_curr = use_ping ? dXb_ping : dXb_pong;
        __nv_bfloat16* dXb_next = use_ping ? dXb_pong : dXb_ping;
        double* dXd_curr = use_ping ? dXd_ping : dXd_pong;
        double* dXd_next = use_ping ? dXd_pong : dXd_ping;

        // Wait for H2D of current chunk to complete before computing
        cuda_check(cudaStreamWaitEvent(stream_compute, h2d_done, 0), "wait h2d_done");

        // Start H2D for next chunk (if any) while computing current
        if (i + 1 < chunks.size()) {
            const auto& next = chunks[i + 1];
            const size_t next_elems = static_cast<size_t>(next.rows) * static_cast<size_t>(N);

            // H2D stream must wait for compute to finish using the next buffer
            cuda_check(cudaStreamWaitEvent(stream_h2d, compute_done, 0), "wait compute_done");

            cudaEventRecord(h2d_start[i + 1], stream_h2d);
            cuda_check(cudaMemcpyAsync(dXf_next, next.src + next.offset * N,
                                       next_elems * sizeof(float),
                                       cudaMemcpyHostToDevice, stream_h2d), "H2D next chunk");
            cudaEventRecord(h2d_stop[i + 1], stream_h2d);
            cudaEventRecord(h2d_done, stream_h2d);
        }

        // Run compute on current chunk
        run_compute(dXf_curr, dXh_curr, dXb_curr, dXd_curr, c.rows, elems, beta, static_cast<int>(i));
        cudaEventRecord(compute_done, stream_compute);

        use_ping = !use_ping;
    }

    // Mirror triangle if using syrk
    if (want_fp32 && mode.algorithm == "syrk") {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        const int upper_from_lower = (uplo == CUBLAS_FILL_MODE_LOWER) ? 1 : 0;
        mirror_triangle<<<grid, block, 0, stream_compute>>>(dC, N, upper_from_lower);
        cuda_check(cudaGetLastError(), "mirror_triangle kernel");
    }

    // Cast fp64 output to fp32 for storage
    if (want_fp64) {
        size_t C_elems = static_cast<size_t>(N) * static_cast<size_t>(N);
        int threads = 256;
        int blocks = static_cast<int>((C_elems + threads - 1) / threads);
        cast_f64_to_f32<<<blocks, threads, 0, stream_compute>>>(dC_fp64, dC, C_elems);
        cuda_check(cudaGetLastError(), "cast fp64 to fp32 output kernel");
    }

    // copy back if required
    if (copy_back) {
        cuda_check(cudaMemcpyAsync(C_out_row_major, dC, bytes_C,
                                   cudaMemcpyDeviceToHost, stream_compute), "D2H result");
    }
    cuda_check(cudaStreamSynchronize(stream_compute), "sync compute stream");
    cuda_check(cudaStreamSynchronize(stream_h2d), "sync h2d stream");

    // Timing
    float gemm_ms_total = 0.0f;
    float h2d_ms_total = 0.0f;
    float cast_ms_total = 0.0f;
    for (size_t i = 0; i < chunks.size(); ++i) {
        float gemm = 0.f;
        cudaEventElapsedTime(&gemm, gemm_start[i], gemm_stop[i]);
        gemm_ms_total += gemm;

        float h2d = 0.f;
        cudaEventElapsedTime(&h2d, h2d_start[i], h2d_stop[i]);
        h2d_ms_total += h2d;

        if (!want_fp32 && !want_tf32) {
            float cast = 0.f;
            cudaEventElapsedTime(&cast, cast_start[i], cast_stop[i]);
            cast_ms_total += cast;
        }
    }
    time.gemm_ms = gemm_ms_total;
    time.h2d_ms  = h2d_ms_total;
    time.cast_ms = cast_ms_total;
}

static void compute_xtx_single_device(
        const ComputeParams& params, int device_id,
        const float *X_local, std::int64_t rows_local,
        const float *X_remote, std::int64_t rows_remote,
        float *C_out_row_major, bool copy_back,
        GpuBuffers& bufs,
        GpuTimeTotal& time
        ) {
    const ModeCfg& mode = params.mode;
    const ComputeScalars& scalars = params.scalars;
    const int N = static_cast<int> (params.N);

    if (!C_out_row_major) {
        throw std::runtime_error("C_out_row_major must not be null");
    }

    cuda_check(cudaSetDevice(device_id), "CudaSetDevice");

    // Use pre-allocated stream (use stream_compute as main stream for single buffering)
    cudaStream_t stream = bufs.stream_compute;
    cublasHandle_t handle = bufs.cublas_handle;

    // Set math mode (needs to be done each call as mode might change)
    cublasMath_t math_mode = parse_math_mode(mode.cublas_math_mode);
    cublas_check(cublasSetMathMode(handle, math_mode), "cublasSetMathMode");

    // Use pre-allocated buffers
    float* dC = bufs.dC;
    size_t bytesC = bufs.bytes_C;
    bufs.reset_output(stream);

    const int64_t rows_per_chunk = params.chunking.rows_per_chunk;

    // Single buffer for input (pre-allocated as dX_ping)
    float* dXf = bufs.dX_ping;

    // Casted buffers (pre-allocated)
    __half* dXh = bufs.dXh_ping;
    __nv_bfloat16* dXb = bufs.dXb_ping;
    double* dXd = bufs.dXd_ping;
    double* dC_fp64 = bufs.dC_fp64;

    // compute type
    const std::string& dtype = mode.name;

    const bool want_fp16 = (dtype == "fp16");
    const bool want_bf16 = (dtype == "bf16");
    const bool want_tf32 = (dtype == "tf32");
    const bool want_fp64 = (dtype == "fp64");
    const bool want_fp32 = (!want_fp16 && !want_bf16 && !want_tf32 && !want_fp64);

    const cublasFillMode_t uplo = parse_triangle(mode.triangle);

    // Use pre-allocated events
    std::vector<cudaEvent_t>& gemm_start = bufs.gemm_start;
    std::vector<cudaEvent_t>& gemm_stop = bufs.gemm_stop;
    std::vector<cudaEvent_t>& h2d_start = bufs.h2d_start;
    std::vector<cudaEvent_t>& h2d_stop = bufs.h2d_stop;
    std::vector<cudaEvent_t>& cast_start = bufs.cast_start;
    std::vector<cudaEvent_t>& cast_stop = bufs.cast_stop;

    int chunk_id = 0;

    auto process_source = [&](const float* X, int64_t rows) {
        int64_t done = 0;
        while (done < rows) {
            int64_t K = std::min<int64_t> (rows_per_chunk, rows - done);
            const size_t elems = static_cast<size_t> (K) * static_cast<size_t> (N);

            // H2D copy
            cudaEventRecord(h2d_start[chunk_id], stream);
            cuda_check(
                    cudaMemcpyAsync(dXf, X + static_cast<size_t> (done) * N, 
                                    elems * sizeof(float), 
                                    cudaMemcpyHostToDevice, stream
                                    ), 
                    "cudaMemcpuAsync dXf -> X"
            );
            cudaEventRecord(h2d_stop[chunk_id], stream);

            const float alpha = scalars.alpha;
            const float beta  = (done == 0 && X == X_local) ? scalars.beta_first : scalars.beta_rest;


            if (want_fp32) {
                cudaEventRecord(gemm_start[chunk_id], stream);
                if (mode.algorithm == "syrk") {
                    run_1_chunk_fp32_syrk(handle, uplo,
                                          N, static_cast<int> (K), 
                                          dXf, dC,
                                          alpha, beta);
                }
                else {
                    if (USE_CUBLASLT_XTX) {
                        run_1_chunk_fp32_xtx_cublaslt(
                            N, static_cast<int>(K),
                            dXf, dC,
                            alpha, beta,
                            stream,
                            /*workspace=*/nullptr,
                            /*workspace_bytes=*/CUBLASLT_WORKSPACE_BYTES,
                            /*enable_tf32=*/CUBLASLT_ENABLE_TF32
                        );
                    } else {
                        run_1_chunk_fp32_gemm(handle,
                                              N, static_cast<int>(K),
                                              dXf, dC,
                                              alpha, beta);
                    }
                }
                cudaEventRecord(gemm_stop[chunk_id], stream);
            }
            else if (want_tf32) {
                cudaEventRecord(gemm_start[chunk_id], stream);
                if (mode.algorithm == "syrk") {
                    
                    run_1_chunk_fp32_syrk(handle, uplo,
                                          N, static_cast<int>(K),
                                          dXf, dC,
                                          alpha, beta);
                }
                else {
                    if (USE_CUBLASLT_XTX) {
                        run_1_chunk_fp32_xtx_cublaslt(
                            N, static_cast<int>(K),
                            dXf, dC,
                            alpha, beta,
                            stream,
                            /*workspace=*/nullptr,
                            /*workspace_bytes=*/CUBLASLT_WORKSPACE_BYTES,
                            /*enable_tf32=*/true
                        );
                    } 
                    else {                   // TF32 Tensor core compute, FP32 accumulate
                        run_1_chunk_gemm_ex(handle,
                                            N, static_cast<int>(K),
                                            dXf, CUDA_R_32F,
                                            dC,
                                            alpha, beta,
                                            CUBLAS_COMPUTE_32F_FAST_TF32);
                    }
                }
                cudaEventRecord(gemm_stop[chunk_id], stream);
            }
            else if (want_fp64) {            // FP64 compute, FP64 accumulate
                int thread = 256;
                int blocks = static_cast<int>((elems + thread - 1) / thread);

                cudaEventRecord(cast_start[chunk_id], stream);
                cast_f32_to<<<blocks, thread, 0, stream>>>(dXf, dXd, elems);
                cuda_check(cudaGetLastError(), "cast fp32 to fp64 kernel");
                cudaEventRecord(cast_stop[chunk_id], stream);

                cudaEventRecord(gemm_start[chunk_id], stream);
                run_1_chunk_fp64_gemm(handle,
                                      N, static_cast<int>(K),
                                      dXd, dC_fp64,
                                      static_cast<double>(alpha), static_cast<double>(beta));
                cudaEventRecord(gemm_stop[chunk_id], stream);
            }                               // FP 16 Tensor core compute, FP32 accumulate
            else if (want_fp16) {
                int thread = 256;
                int blocks = static_cast<int>((elems + thread - 1) / thread);

                cudaEventRecord(cast_start[chunk_id], stream);
                cast_f32_to<<<blocks, thread, 0, stream>>>(dXf, dXh, elems);
                cuda_check(cudaGetLastError(), "cast fp32 to fp16 kernel");
                cudaEventRecord(cast_stop[chunk_id], stream);

                cudaEventRecord(gemm_start[chunk_id], stream);
                run_1_chunk_gemm_ex(handle,
                                    N, static_cast<int> (K),
                                    dXh, CUDA_R_16F,
                                    dC,
                                    alpha, beta,
                                    CUBLAS_COMPUTE_32F);
                cudaEventRecord(gemm_stop[chunk_id], stream);
            }
            else {                          // BF16 Tensor core compute, FP32 accumulate
                int thread = 256;
                int blocks = static_cast<int>((elems + thread - 1) / thread);

                cudaEventRecord(cast_start[chunk_id], stream);
                cast_f32_to<<<blocks, thread, 0, stream>>>(dXf, dXb, elems);
                cuda_check(cudaGetLastError(), "cast fp32 to bf16 kernel");
                cudaEventRecord(cast_stop[chunk_id], stream);

                cudaEventRecord(gemm_start[chunk_id], stream);
                run_1_chunk_gemm_ex(handle,
                                    N, static_cast<int> (K),
                                    dXb, CUDA_R_16BF,
                                    dC, 
                                    alpha, beta,
                                    CUBLAS_COMPUTE_32F);
                cudaEventRecord(gemm_stop[chunk_id], stream);
            }
            
            done += K;
            chunk_id++;
        }
    };

    if (X_local && rows_local > 0)  process_source(X_local, rows_local);
    if (X_remote && rows_remote > 0) process_source(X_remote, rows_remote);


    if (want_fp32 && mode.algorithm == "syrk") {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        const int upper_from_lower = (uplo == CUBLAS_FILL_MODE_LOWER) ? 1 : 0;
        mirror_triangle<<<grid, block, 0, stream>>>(dC, N, upper_from_lower);
        cuda_check(cudaGetLastError(), "mirror_triangle kernel");
    }

    // Cast fp64 output to fp32 for storage
    if (want_fp64) {
        size_t C_elems = static_cast<size_t>(N) * static_cast<size_t>(N);
        int threads = 256;
        int blocks = static_cast<int>((C_elems + threads - 1) / threads);
        cast_f64_to_f32<<<blocks, threads, 0, stream>>>(dC_fp64, dC, C_elems);
        cuda_check(cudaGetLastError(), "cast fp64 to fp32 output kernel");
    }

    if (copy_back) {
        cuda_check(cudaMemcpyAsync(C_out_row_major, dC, bytesC, cudaMemcpyDeviceToHost, stream), "cudaMemcopyAsync device 2 host");
    }
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    // Timing
    float gemm_ms_total = 0.0f;
    float h2d_ms_total = 0.0f;
    float cast_ms_total = 0.0f;
    for (int i = 0; i < chunk_id; ++i) {
        float gemm = 0.f;
        cudaEventElapsedTime(&gemm, gemm_start[i], gemm_stop[i]);
        gemm_ms_total += gemm;

        float h2d = 0.f;
        cudaEventElapsedTime(&h2d, h2d_start[i], h2d_stop[i]);
        h2d_ms_total += h2d;

        if (!want_fp32 && !want_tf32) {
            float cast = 0.f;
            cudaEventElapsedTime(&cast, cast_start[i], cast_stop[i]);
            cast_ms_total += cast;
        }
    }
    time.gemm_ms = gemm_ms_total;
    time.h2d_ms  = h2d_ms_total;
    time.cast_ms = cast_ms_total;
}

std::vector<GpuTimeTotal> compute_xtx_multi_device(
    const ComputeParams& params,
    const GeneratedMatrix& X,
    float* C_out_row_major,
    std::vector<GpuBuffers>& gpu_buffers
) {
    const int used = (int)params.devices.size();
    if (used == 0) throw std::runtime_error("params.devices is empty");

    // times
    std::vector<GpuTimeTotal> times(used);
    
    // only 1 gpu
    if (used == 1) {
        const int device_id = params.devices[0].device_id;   
        const int gpu_node = gpu_to_numa_node(device_id);
        const NodeBuffer* src0 = nullptr;  // primary source
        const NodeBuffer* src1 = nullptr;  // optional second source
        
        // Prefer buffer that matches the GPU NUMA node
        for (const auto& b : X.bufs) {
            if (b.node == gpu_node) { src0 = &b; break; }
        }
        
        // Fallback: if no "local-to-GPU" buffer exists, pick the first buffer
        if (!src0) {
            src0 = &X.bufs[0];
        }
        
        // Pick a second, distinct buffer
        for (const auto& b : X.bufs) {
            if (&b != src0) { src1 = &b; break; }
        }
        
        
        if (params.double_buffering) {
            compute_xtx_double_buffering(
                params, device_id,
                src0->ptr, src0->rows,
                src1 ? src1->ptr : nullptr,
                src1 ? src1->rows : 0,
                C_out_row_major, true,  // copy_back = true for single GPU
                gpu_buffers[0],
                times[0]);
        } else {
            compute_xtx_single_device(
                params, device_id,
                src0->ptr, src0->rows,
                src1 ? src1->ptr : nullptr,
                src1 ? src1->rows : 0,
                C_out_row_major, true,  // copy_back = true for single GPU
                gpu_buffers[0],
                times[0]);
        }
        return times;
    }



    std::vector<std::thread> gpu_threads;
    gpu_threads.reserve(used);

    // else launch exactly the GPUs specified in YAML
    for (int i = 0; i < used; ++i) {
        const int device_id = params.devices[i].device_id;

        gpu_threads.emplace_back([&, i, device_id] {
            // detect which NUMA node this GPU is closest to
            const int gpu_node = gpu_to_numa_node(device_id);

            // pick buffer that lives on that NUMA node
            const NodeBuffer* buf = nullptr;
            for (const auto& b : X.bufs) {
                if (b.node == gpu_node) { buf = &b; break; }
            }
            if (!buf) buf = &X.bufs[0]; // fallback

            // compute partial C on this GPU
            if (params.double_buffering) {
                compute_xtx_double_buffering(
                    params, device_id,
                    buf->ptr, buf->rows,   // X_local
                    nullptr, 0,            // X_remote unused in multi-GPU
                    C_out_row_major, false,
                    gpu_buffers[i],
                    times[i]);
            } else {
                compute_xtx_single_device(
                    params, device_id,
                    buf->ptr, buf->rows,   // X_local
                    nullptr, 0,            // X_remote unused in multi-GPU
                    C_out_row_major, false,
                    gpu_buffers[i],
                    times[i]);
            }
        });
    }

    for (auto& t : gpu_threads) t.join();

    // ---- Parallel D2H transfers ----
    // Each GPU copies its dC to its pinned h_C_partial buffer
    for (int i = 0; i < used; ++i) {
        cuda_check(cudaSetDevice(params.devices[i].device_id), "cudaSetDevice for D2H");
        cuda_check(cudaMemcpyAsync(
            gpu_buffers[i].h_C_partial,
            gpu_buffers[i].dC,
            gpu_buffers[i].bytes_C,
            cudaMemcpyDeviceToHost,
            gpu_buffers[i].stream_compute
        ), "D2H partial result");
    }

    // Sync all D2H transfers
    for (int i = 0; i < used; ++i) {
        cuda_check(cudaSetDevice(params.devices[i].device_id), "cudaSetDevice for sync");
        cuda_check(cudaStreamSynchronize(gpu_buffers[i].stream_compute), "sync D2H");
    }

    // ---- CPU reduction with OpenMP + SIMD ----
    const size_t C_elems = gpu_buffers[0].bytes_C / sizeof(float);

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < C_elems; ++i) {
        float sum = 0.0f;
        for (int g = 0; g < used; ++g) {
            sum += gpu_buffers[g].h_C_partial[i];
        }
        C_out_row_major[i] = sum;
    }

    return times;
}




