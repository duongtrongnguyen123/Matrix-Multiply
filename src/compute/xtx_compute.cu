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


#include <compute/xtx_compute.h>
#include <compute/utils.h>
#include <compute/xtx_cublas.h>
#include <compute/xtx_cublasLt.h>


// ===== quick cublasLt XTX test knobs =====
static constexpr bool USE_CUBLASLT_XTX = true;
static constexpr bool CUBLASLT_ENABLE_TF32 = true;

// 256 MB workspace is safe for most GEMM shapes
static constexpr size_t CUBLASLT_WORKSPACE_BYTES = 256ull * 1024 * 1024;

__global__ void mirror_triangle(float* C, int N, int upper_from_lower) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;
    if (i == j) return;

    if (upper_from_lower) {
        if (i < j) C[i + j * N] = C[j + i * N]; // C[i, j] = C[j, i]
    }
    else {
        if (i > j) C[j + i * N] = C[i + j * N]; // C[j, i] = C[i, j]
    }
}


static void compute_xtx_gpu_single_device(
        const ComputeParams& params, int device_id,
        const float *X_local, std::int64_t rows_local,
        const float *X_remote, std::int64_t rows_remote,
        float *C_out_row_major, bool copy_back,
        GpuTimeTotal& time
        ) {
    const int N = static_cast<int> (params.N);

    if (!C_out_row_major) {
        throw std::runtime_error("C_out_row_major must not be null");
    }

    cuda_check(cudaSetDevice(device_id), "CudaSetDevice");
    
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    cublasHandle_t handle;
    cublas_check(cublasCreate(&handle), "cublasCreate");
    cublas_check(cublasSetStream(handle, stream), "cublasSetStream");

    cublasMath_t math_mode = parse_math_mode(params.cublas_math_mode);
    cublas_check(cublasSetMathMode(handle, math_mode), "cublasSetMathMode");

    // output C on device
    float* dC = nullptr;
    size_t bytesC = static_cast<size_t> (N) *static_cast<size_t> (N) * sizeof(float);
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dC), bytesC), "cudaMalloc dC");
    cuda_check(cudaMemsetAsync(dC, 0, bytesC, stream), "cudaMemsetAsync dC");

    const int64_t rows_per_chunk = params.rows_per_chunk;
    const size_t max_chunk_elems = static_cast<size_t> (rows_per_chunk) * static_cast<size_t> (N);

    float *dXf = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dXf), max_chunk_elems * sizeof(float)), "cudaMalloc");

    __half *dXh = nullptr;
    __nv_bfloat16 *dXb = nullptr;
    

    // compute type
    std::string dtype = params.input_dtype;

    const bool want_fp16 = (dtype == "fp16");
    const bool want_bf16 = (dtype == "bf16");
    const bool want_fp32 = (!want_fp16 && !want_bf16);

    if (want_fp16) cuda_check(
                        cudaMalloc(reinterpret_cast<void**>(&dXh),                        max_chunk_elems * sizeof(__half)), 
                        "cudaMallocfp16");
    if (want_bf16) cuda_check(
                        cudaMalloc(reinterpret_cast<void**>(&dXf),                        max_chunk_elems * sizeof(__nv_bfloat16)),
                        "cudaMallocbf16");
    
    // timer to time H2D time, compute time, casting time
    const cublasFillMode_t uplo = parse_triangle(params.triangle);

    int max_chunks = (rows_local + rows_per_chunk - 1) / rows_per_chunk +
                     (rows_remote + rows_per_chunk - 1) / rows_per_chunk;

    std::vector<cudaEvent_t> gemm_start(max_chunks, 0), gemm_stop(max_chunks, 0);
    std::vector<cudaEvent_t> h2d_start(max_chunks, 0), h2d_stop(max_chunks, 0);
    std::vector<cudaEvent_t> cast_start(max_chunks, 0), cast_stop(max_chunks, 0);

    for (int i = 0; i < max_chunks; ++i) {
        cudaEventCreate(&gemm_start[i]);
        cudaEventCreate(&gemm_stop[i]);
        cudaEventCreate(&h2d_start[i]);
        cudaEventCreate(&h2d_stop[i]);
        cudaEventCreate(&cast_start[i]);
        cudaEventCreate(&cast_stop[i]);
    }

    int chunk_id = 0;

    // X points to DRAM, diff with dXf
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
            // stop H2D timer

            const float alpha = params.alpha;
            const float beta  = (done == 0 && X == X_local) ? params.beta_first : params.beta_rest;

            
            if (want_fp32) {
                
                if (params.algorithm == "syrk") {
                    run_1_chunk_fp32_syrk(handle, uplo,
                                          N, static_cast<int> (K), 
                                          dXf, dC,
                                          alpha, beta);
                }
                else {
                    if (USE_CUBLASLT_XTX) {
                        cudaEventRecord(gemm_start[chunk_id], stream);
                        run_1_chunk_fp32_xtx_cublaslt(
                            N, static_cast<int>(K),
                            dXf, dC,
                            alpha, beta,
                            stream,
                            /*workspace=*/nullptr,
                            /*workspace_bytes=*/CUBLASLT_WORKSPACE_BYTES,
                            /*enable_tf32=*/CUBLASLT_ENABLE_TF32
                        );
                        cudaEventRecord(gemm_stop[chunk_id], stream);
                    } else {
                        
                        run_1_chunk_fp32_gemm(handle,
                                              N, static_cast<int>(K),
                                              dXf, dC,
                                              alpha, beta);
                    }
                }
                
            }
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
            else {
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

    
    if (want_fp32 && params.algorithm == "syrk") {
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        const int upper_from_lower = (uplo == CUBLAS_FILL_MODE_LOWER) ? 1 : 0;
        mirror_triangle<<<grid, block, 0, stream>>>(dC, N, upper_from_lower);
        cuda_check(cudaGetLastError(), "mirror_triangle kernel");
    }

    // storage to host from device
    // interpreting col-major as row-major
    if (copy_back) {
        cuda_check(cudaMemcpyAsync(C_out_row_major, dC, bytesC, cudaMemcpyDeviceToHost, stream), "cudaMemcopyAsync device 2 host");
    }        
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");


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

        if (!want_fp32) {
            float cast = 0.f;
            cudaEventElapsedTime(&cast, cast_start[i], cast_stop[i]);
            cast_ms_total += cast;
        }
    }
    time.gemm_ms = gemm_ms_total;
    time.h2d_ms  = h2d_ms_total;
    time.cast_ms = cast_ms_total;


    if (dXb) cudaFree(dXb);
    if (dXh) cudaFree(dXh);
    cudaFree(dXf);
    cudaFree(dC);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}




std::vector<GpuTimeTotal> compute_xtx_multi_device(
    const ComputeParams& params,
    const GeneratedMatrix& X,
    float* C_out_row_major
) {
    int visible = 0;
    cuda_check(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");

    const int used = (int)params.devices.size();
    if (used == 0) throw std::runtime_error("params.devices is empty");

    // validate YAML device ids exist at runtime
    for (const auto& d : params.devices) {
        if (d.device_id < 0 || d.device_id >= visible) {
            throw std::runtime_error(
                "YAML device_id " + std::to_string(d.device_id) +
                " not visible at runtime (visible=" + std::to_string(visible) + ")"
            );
        }
    }

    // times
    std::vector<GpuTimeTotal> times(used);
    
    // only 1 gpu
    if (used == 1) {
        const int device_id = params.devices[0].device_id;   
        const int gpu_node = gpu_to_numa_node(device_id);
        const NodeBuffer* src0 = nullptr;  // primary source
        const NodeBuffer* src1 = nullptr;  // optional second source
        
        // 1) Prefer a buffer that matches the GPU NUMA node
        for (const auto& b : X.bufs) {
            if (b.node == gpu_node) { src0 = &b; break; }
        }
        
        // 2) Fallback: if no "local-to-GPU" buffer exists, pick the first buffer
        if (!src0) {
            src0 = &X.bufs[0];
        }
        
        // 3) Pick a second, distinct buffer (if any)
        for (const auto& b : X.bufs) {
            if (&b != src0) { src1 = &b; break; }
        }
        
        
        compute_xtx_gpu_single_device(
            params, device_id,
            src0->ptr, src0->rows,
            src1 ? src1->ptr : nullptr,
            src1 ? src1->rows : 0,
            C_out_row_major, false,
            times[0]);
        return times;
    }




    const int64_t N = params.N;
    const size_t C_elems = static_cast<size_t> (N) * static_cast<size_t> (N);

    // one partial C per USED GPU (not per visible GPU)
    std::vector<std::vector<float>> C_partial(
        used, std::vector<float>(C_elems, 0.0f)
    );

    std::vector<std::thread> gpu_threads;
    gpu_threads.reserve(used);

    // else launch exactly the GPUs specified in YAML
    for (int i = 0; i < used; ++i) {
        const int device_id = params.devices[i].device_id;

        gpu_threads.emplace_back([&, i, device_id] {
            // per-thread params copy so we can set device_id without races

            // detect which NUMA node this GPU is closest to
            const int gpu_node = gpu_to_numa_node(device_id);

            // pick buffer that lives on that NUMA node
            const NodeBuffer* buf = nullptr;
            for (const auto& b : X.bufs) {
                if (b.node == gpu_node) { buf = &b; break; }
            }
            if (!buf) buf = &X.bufs[0]; // fallback

            // compute partial C on this GPU
            compute_xtx_gpu_single_device(
                params, device_id,
                buf->ptr, buf->rows,   // X_local
                nullptr, 0,            // X_remote unused in multi-GPU
                C_partial[i].data(), false,
                times[i]);
        });
    }

    for (auto& t : gpu_threads) t.join();

    // reduce partials on CPU
    std::memset(C_out_row_major, 0, C_elems * sizeof(float));
    for (int i = 0; i < used; ++i) {
        for (size_t k = 0; k < C_elems; ++k) {
            C_out_row_major[k] += C_partial[i][k];
        }
    }
    return times;
}




