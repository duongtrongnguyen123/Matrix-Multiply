#include <numa.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cmath>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include <config/config.h>

// hash function
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}


// create a deterministic random float 
static inline float make_randn_float(uint64_t seed, uint64_t idx) { 
    uint64_t r1 = splitmix64(seed ^ (idx * 0x9E3779B97F4A7C15ull));
    uint64_t r2 = splitmix64(seed ^ (idx * 0xBF58476D1CE4E5B9ull));

    // Convert to uniform (0, 1]
    const float inv2p32 = 1.0f / 4294967296.0f;
    float u1 = ((uint32_t)(r1 >> 32) + 1.0f) * inv2p32;
    float u2 = ((uint32_t)(r2 >> 32) + 1.0f) * inv2p32;

    float radius = sqrtf(-2.0f * logf(u1));
    float theta = 6.283185307179586f * u2;

    return radius * cosf(theta);
}


// allocate matrix
static void* numa_alloc_or_throw(size_t bytes, int node) {
    void *p = numa_alloc_onnode(bytes, node);
    
    if (!p) {
        throw std::runtime_error("numa_alloc_on_node fail for " + std::to_string(bytes) + " on node " + std::to_string(node));
    }

    return p;
}

// fill matrix
static void fill_matrix_row_major(float* X, int64_t rows, int64_t cols, uint64_t seed, uint64_t global_row_offset) {
    // Fill rows * cols matrix deterministic via seed, global index
    // global_row_offset for not duplicate rows index in different numa node
    const uint64_t N = static_cast<uint64_t>(cols);

    #pragma omp parallel for schedule(static)
    for (int i = 0 ; i < rows ; i ++ ) {
        uint64_t grow = static_cast<uint64_t>(static_cast<uint64_t>(i) + global_row_offset);
        uint64_t base = grow * N; // each row has unique base, even in different numa node
        float *rowp = X + static_cast<size_t>(i) * static_cast<size_t>(cols);
        for (int j = 0 ; j < cols ; j ++ ) {
           uint64_t idx = static_cast<uint64_t>(j) + base;
           rowp[j] = make_randn_float(seed, idx);
        }
    }
}

static void generate_random_matrix_2_numa(
        Config& cfg,
        float** X_local, int64_t* rows_local,
        float** X_remote, int64_t* rows_remote
        ) {
    if (numa_available() < 0) {
        throw std::runtime_error("numa not support on this device numa_available() < 0");
    }

    const int max_node = numa_max_node();
    if (cfg.gpu_local_node < 0 || cfg.gpu_local_node > max_node 
        || cfg.remote_mode < 0 || cfg.remote_node > max_node) {
        throw std::runtime_error("invalid local_node or remote_node");
    }

    int64_t m_local = static_cast<int64_t>(cfg.M) * cfg.split_ratio;
    int64_t m_remote = cfg.M - m_local;

    const size_t bytes_local = static_cast<size_t> (m_local) * cfg.N * sizeof(float);
    const size_t bytes_remote = static_cast<size_t>(m_remote) * cfg.N * sizeof(float);

    std::cout << "gen M: " << cfg.M << "N: " << cfg.N << endl;
    std::cout << "split_ratio: " << cfg.split_ratio << " ||| local_rows: " << m_local << " ||| remote_rows: " << m_remote << endl;
    std::cout << "gen bytes_local=" << (bytes_local / (1024.0*1024.0*1024.0)) << " GiB"
              << " bytes_remote=" << (bytes_remote / (1024.0*1024.0*1024.0)) << " GiB\n";

    float *xl = static_cast<float*>(numa_alloc_or_throw(bytes_local, cfg.gpu_local_node));
    float *xr = static_cast<float*>(numa_alloc_or_throw(bytes_remote, cfg.remote_node));

    fill_matrix_row_major(xl, m_local, cfg.N, cfg.seed, 0);
    fill_matrix_row_major(xr, m_remote, cfg.N, cfg.seed, static_cast<uint64_t>(m_local));

    *X_local = xl;
    *X_remote = xr;
    *rows_local = m_local;
    *rows_remote = m_remote;
    
    std::cout << "done" << std::endl;
}

static void free_2_numa(float* X_local, size_t bytes_local, float* X_remote, size_t bytes_remote) {
    if (X_local) {
        numa_free(X_local, bytes_local);
    }
    if (X_remote) {
        numa_free(X_remote, bytes_remote);
    }
}

int main() {
    try {
        std::string path = "/../config/xtx_precision_perf.yaml";
        if (argc >= 2) path = args[1];

        Config cfg = load_config_yaml(path);
        std::cout << "[cfg] config name: " << cfg.name << " ||| "                       << "seed: " << cfg.seed << " ||| "                          << "numa local node: " << cfg.gpu_local_node << " ||| "       << "numa remote node: " << cfg.remote_node << std::endl;

#ifdef _OPENMP
    std::cout << "[omp] max_threads=" << omp_get_max_threads() << "\n";
#else
    std::cout << "[omp] disabled (compile with -Xcompiler -fopenmp)\n";
#endif

    float* X_local = nullptr;
    float* X_remote = nullptr;
    uint64_t rows_local = 0;
    uint64_t rows_remote = 0;

    generate_random_matrix_2_numa(cfg, X_local, X_remote, rows_local, rows_remote);
    std::cout << "[check] X_local[0,0]=" << X_local[0] << "  X_local[0,1]=" << X_local[1] << "\n";
    std::cout << "[check] X_remote[0,0]=" << X_remote[0] << "  X_remote[0,1]=" << X_remote[1] << "\n";
    
    size_t bytes_local  = static_cast<size_t>(rows_local)  * cfg.N * sizeof(float);
    size_t bytes_remote = static_cast<size_t>(rows_remote) * cfg.N * sizeof(float);
    free_two_numa(X_local, bytes_local, X_remote, bytes_remote);
    } catch (const std::exception& e){
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
    }
}
