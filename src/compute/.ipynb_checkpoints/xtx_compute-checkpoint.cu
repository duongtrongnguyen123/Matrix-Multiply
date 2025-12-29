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

static inline std::string lower(std::string x) {
    for (char& c : x) c = (char)std::tolower((unsigned char)c);
    return x;
}

static inline cublasMath_t parse_math_mode(std::string s) {
    if (s.empty()) return CUBLAS_DEFAULT_MATH;
    s = lower(s);

    if (s == "default") return CUBLAS_DEFAULT_MATH;
    if (s == "tf32") return CUBLAS_TF32_TENSOR_OP_MATH;
    if (s == "pedantic") return CUBLAS_PEDANTIC_MATH;

    if (s == "tensor_op" || s == "tensorop" || s == "tensor") return CUBLAS_DEFAULT_MATH;

#if defined(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
    if (s == "disallow_reduced_precision_reduction") {
        return CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
    }
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
        if (i < j) C[i + j * N] = C[j + i * N]; // C[i, j] = C[j, i]
    }
    else {
        if (i > j) C[j + i * N] = C[i + j * N]; // C[j, i] = C[i, j]
    }
}

static int read_int_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return -1;
    int x = -1;
    f >> x;
    return x;
}

static int gpu_to_numa_node(int device_id) {
    char pci[32];
    cudaDeviceGetPCIBusId(pci, sizeof(pci), device_id);
    // pci looks like "0000:65:00.0"

    std::string path = std::string("/sys/bus/pci/devices/") + pci + "/numa_node";
    int node = read_int_file(path);

    if (node < 0) node = 0; // fallback
    return node;
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

template <typename TO>
__global__ void cast_f32_to(const float *in, TO *out, size_t n) {
    uint64_t idx = static_cast<uint64_t> (blockIdx.x) * static_cast<uint64_t> (blockDim.x) + static_cast<uint64_t> (threadIdx.x);
    if (idx < n) out[idx] = static_cast<TO> (in[idx]);
}

static void run_1_chunk_gemm_ex(
        cublasHandle_t h, 
        int N, int K,
        const void *dX, cudaDataType Atype,  // X: k*n
        float *dC,                            // output
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
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP
                          ),
            "cublasSgemmEx"
            );
}

static void compute_xtx_gpu_mode(
        const ComputeParams& params, int device_id,
        const float *X_local, std::int64_t rows_local,
        const float *X_remote, std::int64_t rows_remote,
        float *C_out_row_major, bool copy_back
        ) {
    const int m_total = rows_local + rows_remote;
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

    const cublasFillMode_t uplo = parse_triangle(params.triangle);

    // X points to DRAM, diff with dXf
    auto process_source = [&](const float* X, int64_t rows) {
        int64_t done = 0;
        while (done < rows) {
            int64_t K = std::min<int64_t> (rows_per_chunk, rows-done);
            const size_t elems = static_cast<size_t> (K) * static_cast<size_t> (N);

            // H2D copy
            cuda_check(
                    cudaMemcpyAsync(dXf, X + static_cast<size_t> (done) * N, 
                                    elems * sizeof(float), 
                                    cudaMemcpyHostToDevice, stream
                                    ), 
                    "cudaMemcpuAsync dXf -> X"
            );

            const float alpha = params.alpha;
            const float beta  = (done == 0 && X == X_local) ? params.beta_first : params.beta_rest;

            if (want_fp32) {
                if (params.algorithm == "syrk") {
                    run_1_chunk_fp32_syrk(handle, uplo,
                                          N, static_cast<int> (K), 
                                          dXf, dC,
                                          alpha, beta);
                }
                else run_1_chunk_fp32_gemm(handle, 
                                           N, static_cast<int> (K),
                                           dXf, dC,
                                           alpha, beta); 
            }
            else if (want_fp16) {
                int thread = 256;
                int blocks = static_cast<int>((elems + thread - 1) / thread);
                cast_f32_to<<<blocks, thread, 0, stream>>>(dXf, dXh, elems);
                cuda_check(cudaGetLastError(), "cast fp32 to fp16 kernel");
                run_1_chunk_gemm_ex(handle, 
                                    N, static_cast<int> (K),
                                    dXh, CUDA_R_16F, 
                                    dC, 
                                    alpha, beta, 
                                    CUBLAS_COMPUTE_32F);
            }
            else {
                int thread = 256;
                int blocks = static_cast<int>((elems + thread - 1) / thread);
                cast_f32_to<<<blocks, thread, 0, stream>>>(dXf, dXb, elems);
                cuda_check(cudaGetLastError(), "cast fp32 to bf16 kernel");
                run_1_chunk_gemm_ex(handle,
                                    N, static_cast<int> (K),
                                    dXb, CUDA_R_16BF,
                                    dC, 
                                    alpha, beta,
                                    CUBLAS_COMPUTE_32F);
            }
            
            done += K;
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

    if (dXb) cudaFree(dXb);
    if (dXh) cudaFree(dXh);
    cudaFree(dXf);
    cudaFree(dC);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
}




void compute_xtx_multi_gpu(
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



    // only 1 gpu
    if (used == 1) {
        const int dev_id = params.devices[0].device_id;
        ComputeParams p = params;

        const int gpu_node = gpu_to_numa_node(dev_id);

        const NodeBuffer* local  = nullptr;
        const NodeBuffer* remote = nullptr;

        for (const auto& b : X.bufs) {
            if (b.node == gpu_node && !local) local = &b;
            else if (!remote) remote = &b;
        }
        if (!local)  local  = &X.bufs[0];
        if (!remote) remote = nullptr;

        compute_xtx_gpu_mode(
            p, dev_id,
            local->ptr,  local->rows,
            remote ? remote->ptr : 0,
            remote ? remote->rows : 0,
            C_out_row_major,
            true);
        return;
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
        const int dev_id = params.devices[i].device_id;

        gpu_threads.emplace_back([&, i, dev_id] {
            // per-thread params copy so we can set device_id without races
            ComputeParams p = params;               // <-- important
            // (optional) p.use_streams etc from params.devices[i] if you need

            // detect which NUMA node this GPU is closest to
            const int gpu_node = gpu_to_numa_node(dev_id);

            // pick buffer that lives on that NUMA node
            const NodeBuffer* buf = nullptr;
            for (const auto& b : X.bufs) {
                if (b.node == gpu_node) { buf = &b; break; }
            }
            if (!buf) buf = &X.bufs[0]; // fallback

            // compute partial C on this GPU
            compute_xtx_gpu_mode(
                p, dev_id,
                buf->ptr, buf->rows,   // X_local
                nullptr, 0,            // X_remote unused in multi-GPU
                C_partial[i].data(),
                true                   // copy_back=true for runnable correctness
            );
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
}




