#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

#include <config/xtx_config.h>
#include <generate/xtx_generate.h>
#include <compute/xtx_compute.h>
#include <io/npy_save.h>

static inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static void print_config(const Config& cfg) {
    std::cout << "======Config======" << std::endl;
    std::cout  << "name         " << cfg.name << std::endl;
    std::cout  << "seed         " << cfg.seed << std::endl;
    std::cout  << "device_id    " << cfg.device_id << std::endl;
    std::cout  << "M            " << cfg.M << std::endl;
    std::cout  << "N            " << cfg.N << std::endl;
    std::cout  << "layout       " << cfg.layout << std::endl;
    std::cout << "rows per chunk " << cfg.rows_per_chunk << std::endl;
    std::cout  << "algorithm    " << cfg.algorithm << std::endl;
    std::cout  << "triangle     " << cfg.triangle << std::endl;
    std::cout  << "modes size   " << cfg.modes.size() << std::endl;
    std::cout  << "repeats      " << cfg.repeats << std::endl;
    
    for (size_t i = 0 ; i < cfg.modes.size() ; i ++ ) {
        std::cout << " | input dtype = " << cfg.modes[i].input_dtype << std::endl;
        std::cout << " | math = " << cfg.modes[i].cublas_math_mode << std::endl;
        std::cout << " | accumulate = " << cfg.modes[i].accumulate << std::endl;
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

        GenerateParams gp{cfg.M, cfg.N, 
                        cfg.seed, 
                        cfg.gpu_local_node, cfg.remote_node, 
                        cfg.split_ratio};
        ComputeParams  cp{cfg.N, 
                        cfg.rows_per_chunk, 
                        cfg.device_id,  
                        mode.input_dtype,
                        mode.compute,
                        mode.accumulate,
                        mode.cublas_math_mode,
                        cfg.algorithm, 
                        cfg.triangle, 
                        cfg.alpha, 
                        cfg.beta_first, cfg.beta_rest};


        float* X_local = nullptr;
        float* X_remote = nullptr;
        int64_t rows_local = 0;
        int64_t rows_remote = 0;
        
        generate_random_matrix_2_numa(gp,
                               &X_local, &rows_local, 
                               &X_remote, &rows_remote);
        
        if (!X_local || rows_local < 0) { 
            throw std::runtime_error("generate_matrix_2_numa fail: X_local is null or rows_local < 0");
        }
        if (rows_local + rows_remote != cfg.M) {
            std::cerr << "[warn] rows_local + rows_remote != cfg.M" << rows_local << "      " << rows_remote << std::endl;
        }

        
        std::vector<float> C((size_t) cfg.N * cfg.N, 0.0f);

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
       if (!save_npy_fp32(out_path, C.data(), cfg.N)) {
           std::cerr << "[warn] failed to save npy to: " << out_path << "\n";
       } 
       else {
           std::cout << "Saved C to: " << out_path << "\n";
       }


        const size_t bytes_local = static_cast<size_t> (rows_local) * static_cast<size_t> (cfg.N) * sizeof(float);
        const size_t bytes_remote = static_cast<size_t> (rows_remote) * static_cast<size_t> (cfg.N) * sizeof(float);
        
        free_2_numa(X_local, bytes_local, X_remote, bytes_remote);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << std::endl;
        return 2;
    }
}

