#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>

#include <config/xtx_config.h>
#include <generate/xtx_generate.h>
#include <compute/xtx_compute.h>
#include <io/npy_save.h>

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("[cuda] ") + msg + ": " + cudaGetErrorString(e));
    }
}

static inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
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

    const int64_t N = params.N;
    const size_t C_elems = (size_t)N * (size_t)N;

    // one partial C per USED GPU (not per visible GPU)
    std::vector<std::vector<float>> C_partial(
        used, std::vector<float>(C_elems, 0.0f)
    );

    std::vector<std::thread> gpu_threads;
    gpu_threads.reserve(used);

    // launch exactly the GPUs specified in YAML
    for (int i = 0; i < used; ++i) {
        const int dev_id = params.devices[i].device_id;

        gpu_threads.emplace_back([&, i, dev_id] {
            // per-thread params copy so we can set device_id without races
            ComputeParams p = params;
            p.device_id = dev_id;                // <-- IMPORTANT
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
                p,
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


static void print_config(const Config& cfg) {
    std::cout << "======Config======" << std::endl;
    std::cout  << "name         " << cfg.name << std::endl;
    std::cout  << "seed         " << cfg.matrix.seed << std::endl;
    std::cout  << "device_id    " << cfg.devices[0].device_id << std::endl;
    std::cout  << "M            " << cfg.matrix.M << std::endl;
    std::cout  << "N            " << cfg.matrix.N << std::endl;
    std::cout  << "layout       " << cfg.matrix.layout << std::endl;
    std::cout << "rows per chunk " << cfg.chunking.rows_per_chunk << std::endl;
    std::cout  << "modes size   " << cfg.modes.size() << std::endl;

    std::cout  << "algorithm    " << cfg.modes[0].algorithm << std::endl;
    //std::cout  << "triangle     " << cfg.triangle << std::endl;
    std::cout  << "repeats      " << cfg.benchmark.repeats << std::endl;
    
    for (size_t i = 0 ; i < cfg.modes.size() ; i ++ ) {
        std::cout << " | input dtype = " << cfg.modes[i].input_dtype << std::endl;
        std::cout << " | math = " << cfg.modes[i].cublas_math_mode << std::endl;
        std::cout << " | algorithm = " << cfg.modes[i].algorithm << std::endl;
        std::cout << " | accumulate = " << cfg.modes[i].accumulate << std::endl;
        std::cout << std::endl;
    }
    std::cout << "======end======\n\n";
}



int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " config.yaml [out.npy]\n";
        return 1;
    }

    std::string cfg_path = argv[1];
    std::string out_path = (argc >= 3) ? argv[2] : "";
    try {
        Config cfg = load_config_yaml(cfg_path);
        print_config(cfg);
        ModeCfg mode = cfg.modes[0];

        GenerateParams gp{cfg.matrix.M, cfg.matrix.N, 
                        cfg.matrix.seed, 
                        cfg.host_memory.placement,
                        cfg.host_memory.threads_per_node,
                        cfg.host_memory.max_threads,
                        cfg.host_memory.pin_threads,
                        cfg.host_memory.numa_aware};
        ComputeParams  cp{cfg.matrix.N, 
                        cfg.chunking.rows_per_chunk, 
                        cfg.devices,  
                        mode.input_dtype,
                        mode.algorithm, 
                        mode.compute,
                        mode.cublas_math_mode,
                        mode.accumulate,

                        cfg.compute_scalars.alpha, 
                        cfg.compute_scalars.beta_first, cfg.compute_scalars.beta_rest};


        float* X_local = nullptr;
        float* X_remote = nullptr;
        int64_t rows_local = 0;
        int64_t rows_remote = 0;
        
        GeneratedMatrix matrix = generate_random_matrix_multi_numa(gp);

        
        if (!X_local || rows_local < 0) { 
            throw std::runtime_error("generate_matrix_2_numa fail: X_local is null or rows_local < 0");
        }
        if (rows_local + rows_remote != cfg.M) {
            std::cerr << "[warn] rows_local + rows_remote != cfg.M" << rows_local << "      " << rows_remote << std::endl;
        }

        
        std::vector<float> C(static_cast<size_t> (cfg.matrix.N) * cfg.matrix.N, 0.0f);

        for (int w = 0 ; w < cfg.warmup_iters ; w ++ ) {
            std::fill(C.begin(), C.end(), 0.0f);
            compute_xtx_gpu_mode(cp, 
                            X_local, rows_local,
                            X_remote, rows_remote,
                            C.data()); 
        }

        std::vector<double> times;
        times.reserve(std::max(1, cfg.repeats));

        for (int i = 0 ; i < std::max(1, cfg.repeats) ; i ++ ) {
            std::fill(C.begin(), C.end(), 0.0f);

            const double t0 = now_sec();
            compute_xtx_gpu_mode(cp, 
                            X_local, rows_local,
                            X_remote, rows_remote,
                            C.data()); 
            const double t1 = now_sec();
            const double dt = t1 - t0;
            times.push_back(dt);

            const double FLOPS = 2.0 *  static_cast<double> (cfg.N) * static_cast<double> (cfg.N) * static_cast<double> (cfg.M);
            const double GFLOPS = FLOPS / 1e9;
            const double GFLOPS_sec = GFLOPS / dt;

       }
       std::sort(times.begin(), times.end());
       const double t_min = times.front();
       const double t_med = times[times.size() / 2];
       const double t_max = times.back();
       double t_mean = 0.0;
       for (double x : times) t_mean += x;
       t_mean /= (double)times.size();

       std::cout << "\n===== Summary =====\n";
       std::cout << "repeats: " << times.size() << "\n";
       std::cout << "min    : " << t_min * 1e3 << " ms\n";
       std::cout << "median : " << t_med * 1e3 << " ms\n";
       std::cout << "mean   : " << t_mean * 1e3 << " ms\n";
       std::cout << "max    : " << t_max * 1e3 << " ms\n";
    
       {
           const double FLOPS = 2.0 * static_cast<double> (cfg.M) * static_cast<double> (cfg.N) * static_cast<double> (cfg.N);
           const double GFLOPS = FLOPS / 1e9;
           std::cout << "approx GFLOP/s (min time): " << (GFLOPS / t_min) << "\n";
           std::cout << "approx GFLOP/s (median)  : " << (GFLOPS / t_med) << "\n";
       }

        
       std::cout << "===================\n\n";

       // 6) Save result for further calculations
       if (!save_npy_fp32(out_path, C.data(), cfg.matrix.N)) {
           std::cerr << "[warn] failed to save npy to: " << out_path << "\n";
       } 
       else {
           std::cout << "Saved C to: " << out_path << "\n";
       }


        const size_t bytes_local = static_cast<size_t> (rows_local) * static_cast<size_t> (cfg.matrix.N) * sizeof(float);
        const size_t bytes_remote = static_cast<size_t> (rows_remote) * static_cast<size_t> (cfg.matrix.N) * sizeof(float);
        
        free_2_numa(X_local, bytes_local, X_remote, bytes_remote);

        return 0;

    }
    catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << std::endl;
        return 2;
    }

}

