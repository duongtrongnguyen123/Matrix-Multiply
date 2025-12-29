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


static inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
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
                        mode.triangle,
                        mode.compute,
                        mode.cublas_math_mode,
                        mode.accumulate,

                        cfg.compute_scalars.alpha, 
                        cfg.compute_scalars.beta_first, cfg.compute_scalars.beta_rest};


        GeneratedMatrix X = generate_random_matrix_multi_numa(gp);

        
        std::vector<float> C(static_cast<size_t> (cfg.matrix.N) * cfg.matrix.N, 0.0f);

        for (int w = 0 ; w < cfg.benchmark.warmup_iters ; w ++ ) {
            std::fill(C.begin(), C.end(), 0.0f);
            compute_xtx_multi_gpu(cp, 
                            X,
                            C.data()); 
        }

        std::vector<double> times;
        times.reserve(std::max(1, cfg.benchmark.repeats));

        for (int i = 0 ; i < std::max(1, cfg.benchmark.repeats) ; i ++ ) {
            std::fill(C.begin(), C.end(), 0.0f);

            const double t0 = now_sec();
            compute_xtx_multi_gpu(cp, 
                            X,
                            C.data()); 
            const double t1 = now_sec();
            const double dt = t1 - t0;
            times.push_back(dt);

            const double FLOPS = 2.0 *  static_cast<double> (cfg.matrix.N) * static_cast<double> (cfg.matrix.N) * static_cast<double> (cfg.matrix.M);
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
           const double FLOPS = 2.0 * static_cast<double> (cfg.matrix.M) * static_cast<double> (cfg.matrix.N) * static_cast<double> (cfg.matrix.N);
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


        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << std::endl;
        return 2;
    }

}


