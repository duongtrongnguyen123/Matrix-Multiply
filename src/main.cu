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
#include <cuda_profiler_api.h>
#include <fstream>
#include <stdexcept>
#include <tuple>

#include <config/xtx_config.h>
#include <generate/xtx_generate.h>
#include <compute/xtx_compute.h>
#include <io/npy_save.h>


static inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}





static void print_config(const Config& cfg, const ModeCfg& mode) {
    std::cout << "======Config======" << std::endl;
    std::cout  << "name         " << cfg.name << std::endl;
    std::cout  << "seed         " << cfg.matrix.seed << std::endl;
    std::cout  << "device_id    " << cfg.devices[0].device_id << std::endl;
    std::cout  << "M            " << cfg.matrix.M << std::endl;
    std::cout  << "N            " << cfg.matrix.N << std::endl;
    std::cout  << "layout       " << cfg.matrix.layout << std::endl;
    std::cout  << "rows per chunk " << cfg.chunking.rows_per_chunk << std::endl;
    std::cout  << "modes size   " << cfg.modes.size() << std::endl;

    std::cout  << "algorithm    " << mode.algorithm << std::endl;
    std::cout  << "name         " << mode.name << std::endl;
    //std::cout  << "triangle     " << cfg.triangle << std::endl;
    std::cout  << "repeats      " << cfg.benchmark.repeats << std::endl;
    std::cout  << "mode_idx     " << cfg.benchmark.mode_idx << std::endl;
    std::cout  << "double_buf   " << (cfg.benchmark.double_buffering ? "true" : "false") << std::endl;
    
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
    // std::string out_path = (argc >= 3) ? argv[2] : " ";
    try {
        Config cfg = load_config_yaml(cfg_path);
        size_t mode_idx = cfg.benchmark.mode_idx;
        ModeCfg mode = cfg.modes[mode_idx];
        print_config(cfg, mode);

        std::vector<double> FLOPSs(cfg.modes.size(), 0);
        for (auto& FLOP:FLOPSs) {
            FLOP = 2.0f
            * (double)cfg.matrix.M
            * (double)cfg.matrix.N
            * (double)cfg.matrix.N;
        }
        FLOPSs[0] = 1.0f
            * (double)cfg.matrix.M
            * (double)cfg.matrix.N
            * (double)(cfg.matrix.N + 1);
        GenerateParams gp{cfg.matrix, cfg.host_memory};
        ComputeParams  cp{cfg.chunking, cfg.devices, mode, cfg.compute_scalars, cfg.matrix.N, cfg.benchmark.double_buffering};


        // Initialize CUDA context before cudaHostRegister is called in threads
        cudaFree(0);

        GeneratedMatrix X = generate_random_matrix_multi_numa(gp);

        size_t C_elems  = static_cast<size_t> (cfg.matrix.N) * static_cast<size_t> (cfg.matrix.N);
        size_t C_bytes  = C_elems * sizeof(float);

        int home_node = /* gpu_node[home_gpu] */ 0;

        float* C = reinterpret_cast<float*> (numa_alloc_onnode(C_bytes, home_node));
        if (!C) throw std::runtime_error("numa_alloc_onnode failed");

        // (optional but recommended) first-touch to really back pages on that node
        #pragma omp parallel
        {
            pin_thread_to_numa_node(home_node);   // if you have it; otherwise omit
            #pragma omp for schedule(static)
            for (size_t k = 0; k < C_elems; ++k) C[k] = 0.0f;
        }

        // ---- Pre-allocate GPU buffers (outside timing) ----
        const size_t num_devices = cfg.devices.size();
        std::vector<GpuBuffers> gpu_buffers(num_devices);
        for (size_t i = 0; i < num_devices; ++i) {
            gpu_buffers[i].allocate(
                cfg.devices[i].device_id,
                cfg.matrix.N,
                cfg.chunking.rows_per_chunk,
                mode.name,
                cfg.benchmark.double_buffering
            );
        }

        // ---- warmup ----
        for (int w = 0; w < cfg.benchmark.warmup_iters; ++w) {
             #pragma omp parallel
            {
                pin_thread_to_numa_node(home_node);
                #pragma omp for schedule(static)
                for (size_t k = 0; k < C_elems; ++k) C[k] = 0.0f;
            }
            (void)compute_xtx_multi_device(cp, X, C, gpu_buffers);
        }
        cudaDeviceSynchronize();

        // ---- profiler start (after warmup) ----
        cudaProfilerStart();

        // ---- benchmark ----
        const int R = std::max(1, cfg.benchmark.repeats);
        
        std::vector<double> times;      // wall-clock seconds
        times.reserve(R);
        
        std::vector<double> gpu_times;  // summed GPU total seconds
        gpu_times.reserve(R);
        
        std::vector<double> h2d_times, cast_times, gemm_times;
        h2d_times.reserve(R);
        cast_times.reserve(R);
        gemm_times.reserve(R);
        
        for (int i = 0; i < R; ++i) {
        
            const double t0 = now_sec();
        
            // returns per-GPU stats
            std::vector<GpuTimeTotal> gpu_stats =
                compute_xtx_multi_device(cp, X, C, gpu_buffers);
        
            const double t1 = now_sec();
            const double wall_dt = t1 - t0;   // seconds
            times.push_back(wall_dt);
        
            // ---- MAX across GPUs (parallel execution time) ----
            double h2d_ms  = 0.0;
            double cast_ms = 0.0;
            double gemm_ms = 0.0;
            double total_elapsed_ms = 0.0;

            for (const auto& g : gpu_stats) {
                h2d_ms  = std::max(h2d_ms,  (double)g.h2d_ms);
                cast_ms = std::max(cast_ms, (double)g.cast_ms);
                gemm_ms = std::max(gemm_ms, (double)g.gemm_ms);
                total_elapsed_ms = std::max(total_elapsed_ms, (double)g.total_elapsed_ms);
            }

            // convert ms -> seconds
            const double h2d_s  = h2d_ms  * 1e-3;
            const double cast_s = cast_ms * 1e-3;
            const double gemm_s = gemm_ms * 1e-3;
            const double elapsed_s = total_elapsed_ms * 1e-3;

            h2d_times.push_back(h2d_s);
            cast_times.push_back(cast_s);
            gemm_times.push_back(gemm_s);

            // Use actual elapsed (accounts for overlap) instead of sum
            gpu_times.push_back(elapsed_s);
        }

        // ---- profiler stop ----
        cudaDeviceSynchronize();
        cudaProfilerStop();

        // ---- summary helpers ----
        auto stats = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            const double mn = v.front();
            const double md = v[v.size() / 2];
            const double mx = v.back();
            double mean = 0.0;
            for (double x : v) mean += x;
            mean /= (double)v.size();
            return std::tuple<double,double,double,double>(mn, md, mean, mx);
        };
        
        auto [t_min, t_med, t_mean, t_max] = stats(times);
        
        auto [gpu_min, gpu_med, gpu_mean, gpu_max] = stats(gpu_times);
        auto [h2d_min, h2d_med, h2d_mean, h2d_max] = stats(h2d_times);
        auto [cast_min, cast_med, cast_mean, cast_max] = stats(cast_times);
        auto [gemm_min, gemm_med, gemm_mean, gemm_max] = stats(gemm_times);
        
        const double FLOPS = FLOPSs[mode_idx];
        const double GFLOPS = FLOPS / 1e9;
        
        // ---- print summary ----
        std::cout << "\n===== Summary =====\n";
        std::cout << "repeats: " << times.size() << "\n\n";
        
        std::cout << "[Wall-clock total]\n";
        std::cout << "min    : " << t_min  * 1e3 << " ms\n";
        std::cout << "median : " << t_med  * 1e3 << " ms\n";
        std::cout << "mean   : " << t_mean * 1e3 << " ms\n";
        std::cout << "max    : " << t_max  * 1e3 << " ms\n";
        
        std::cout << "[GPU total (h2d+cast+gemm)]\n";
        std::cout << "min    : " << gpu_min  * 1e3 << " ms\n";
        std::cout << "median : " << gpu_med  * 1e3 << " ms\n";
        std::cout << "mean   : " << gpu_mean * 1e3 << " ms\n";
        std::cout << "max    : " << gpu_max  * 1e3 << " ms\n";
        std::cout << "approx GFLOP/s (min gpu time): " << (GFLOPS / gemm_min) << "\n";
        std::cout << "approx GFLOP/s (median gpu)  : " << (GFLOPS / gemm_med) << "\n\n";
        
        std::cout << "[GPU breakdown]\n";
        std::cout << "H2D   mean: " << h2d_mean  * 1e3 << " ms (median " << h2d_med  * 1e3 << ")\n";
        std::cout << "CAST  mean: " << cast_mean * 1e3 << " ms (median " << cast_med * 1e3 << ")\n";
        std::cout << "GEMM  mean: " << gemm_mean * 1e3 << " ms (median " << gemm_med * 1e3 << ")\n";
        
        std::cout << "===================\n\n";

        // ---- Free GPU buffers (outside timing) ----
        for (auto& buf : gpu_buffers) {
            buf.free();
        }

        /*
       // 6) Save result for further calculations
       if (!save_npy_fp32(out_path, C.data(), cfg.matrix.N)) {
           std::cerr << "[warn] failed to save npy to: " << out_path << "\n";
       }
       else {
           std::cout << "Saved C to: " << out_path << "\n";
       }
        */

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << std::endl;
        return 2;
    }

}


