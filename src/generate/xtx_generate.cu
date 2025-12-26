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

#include <generate/xtx_generate.h>

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
    float u1 = (static_cast<uint32_t>(r1 >> 32) + 1.0f) * inv2p32;
    float u2 = (static_cast<uint32_t>(r2 >> 32) + 1.0f) * inv2p32;

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

static void pin_thread_to_numa_node(int node) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // cpu mask on current numa node
    bitmask* bm = numa_allocate_cpumask();
    if (!bm) throw std::runtime_error("numa_allocate_cpumask failed");

    if (numa_node_to_cpus(node, bm) != 0) {
        numa_free_cpumask(bm);
        throw std::runtime_error("numa_node_to_cpus failed for node " + std::to_string(node));
    }

    for (unsigned i = 0; i < bm->size; ++i) {
        if (numa_bitmask_isbitset(bm, i)) CPU_SET(i, &cpuset);
    }
    numa_free_cpumask(bm);

    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    if (rc != 0) throw std::runtime_error("pthread_setaffinity_np failed");
}


// Touch pages so physical memory is really placed on that node (first-touch).
static void first_touch_zero(void* p, size_t bytes) {
    constexpr size_t page = 4096;
    volatile char* c = static_cast<volatile char*>(p);
    for (size_t i = 0; i < bytes; i += page) c[i] = 0;
    if (bytes) c[bytes - 1] = 0;
}

// Alloc two buffers in parallel (local/remote)
void alloc_2_numa_parallel(
    size_t bytes_local, int node_local, void** out_local,
    size_t bytes_remote, int node_remote, void** out_remote,
    bool do_first_touch = true
) {
    std::exception_ptr ep1 = nullptr, ep2 = nullptr;

    std::thread t1([&] {
        try {
            pin_thread_to_numa_node(node_local);
            void* p = numa_alloc_or_throw(bytes_local, node_local);
            if (do_first_touch) first_touch_zero(p, bytes_local);
            *out_local = p;
        } catch (...) { ep1 = std::current_exception(); }
    });

    std::thread t2([&] {
        try {
            pin_thread_to_numa_node(node_remote);
            void* p = numa_alloc_or_throw(bytes_remote, node_remote);
            if (do_first_touch) first_touch_zero(p, bytes_remote);
            *out_remote = p;
        } catch (...) { ep2 = std::current_exception(); }
    });

    t1.join();
    t2.join();

    if (ep1) std::rethrow_exception(ep1);
    if (ep2) std::rethrow_exception(ep2);
}


// fill matrix
static void fill_matrix_row_major(float* X, int64_t rows, int64_t cols, uint64_t seed, uint64_t global_row_offset) {
    // Fill rows * cols matrix deterministically via seed, global index
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

void generate_random_matrix_2_numa(
        const GenerateParams& params,
        float** X_local, int64_t* rows_local,
        float** X_remote, int64_t* rows_remote
        ) {
    if (numa_available() < 0) {
        throw std::runtime_error("numa not support on this device numa_available() < 0");
    }

    const int max_node = numa_max_node();
    if (params.gpu_local_node < 0 || params.gpu_local_node > max_node 
        || parmas.remote_mode < 0 || params.remote_node > max_node) {
        throw std::runtime_error("invalid local_node or remote_node");
    }

    int64_t m_local = static_cast<int64_t>(params.M) * params.split_ratio;
    int64_t m_remote = params.M - m_local;

    const size_t bytes_local = static_cast<size_t> (m_local) * params.N * sizeof(float);
    const size_t bytes_remote = static_cast<size_t>(m_remote) * params.N * sizeof(float);

    std::cout << "gen M: " << params.M << "N: " << params.N << endl;
    std::cout << "split_ratio: " << params.split_ratio << " ||| local_rows: " << m_local << " ||| remote_rows: " << m_remote << endl;
    std::cout << "gen bytes_local=" << (bytes_local / (1024.0*1024.0*1024.0)) << " GiB"
              << " bytes_remote=" << (bytes_remote / (1024.0*1024.0*1024.0)) << " GiB\n";

    float *xl = static_cast<float*>(numa_alloc_or_throw(bytes_local, params.gpu_local_node));
    float *xr = static_cast<float*>(numa_alloc_or_throw(bytes_remote, params.remote_node));

    fill_matrix_row_major(xl, m_local, params.N, params.seed, 0);
    fill_matrix_row_major(xr, m_remote, params.N, params.seed, static_cast<uint64_t>(m_local));

    *X_local = xl;
    *X_remote = xr;
    *rows_local = m_local;
    *rows_remote = m_remote;
    
    std::cout << "done" << std::endl;
}

void free_2_numa(float* X_local, size_t bytes_local, float* X_remote, size_t bytes_remote) {
    if (X_local) {
        numa_free(X_local, bytes_local);
    }
    if (X_remote) {
        numa_free(X_remote, bytes_remote);
    }
}





