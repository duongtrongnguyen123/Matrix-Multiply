#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <exception>
#include <algorithm>
#include <cstddef>


#include <cmath>

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "generate/xtx_generate.h"

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


void print_numa_residency(void* ptr, size_t bytes) {
    const int page = 4096;
    int pages = bytes / page;

    std::vector<void*> addrs;
    addrs.reserve((pages + 255) / 256);

    for (int i = 0; i < pages; i += 256) { // sample pages
        addrs.push_back((char*)ptr + (size_t)i * page);
    }

    std::vector<int> status(addrs.size());  // size to match sampled pages

    move_pages(0, addrs.size(), addrs.data(), nullptr, status.data(), 0);

    int n0 = 0, n1 = 0;
    for (int s : status) {
        if (s == 0) n0++;
        if (s == 1) n1++;
    }

    printf("NUMA residency: N0=%d  N1=%d (sampled %zu pages)\n", n0, n1, addrs.size());
}


static std::vector<int64_t> split_rows_by_frac(int64_t M, const std::vector<NodeFrac>& placement) {
    std::vector<int64_t> rows(placement.size(), 0);

    int64_t used = 0;
    for (size_t i = 0 ; i < rows.size() - 1 ; i ++ ) {
        double frac = placement[i].frac;
        int64_t row = static_cast<int64_t>(std::floor(frac * M));
        rows[i] = row;
        used += row;
    }

    rows.back() = M - used;
    return rows;
}

void pin_thread_to_numa_node(int node) {
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

static void fill_rows_f32(float* base, int64_t rows, int64_t N, uint64_t seed,
                          int64_t global_row_start,
                          int threads, bool pin_threads, int node)
{
    auto worker = [&](int tid) {
        if (pin_threads) {
            pin_thread_to_numa_node(node); 
        }
        // block partition rows
        int64_t r0 = (rows * tid) / threads;
        int64_t r1 = (rows * (tid + 1)) / threads;

        for (int64_t r = r0; r < r1; ++r) {
            int64_t gr = global_row_start + r;
            float* rowp = base + r * N;

            for (int64_t c = 0; c < N; ++c) {
                uint64_t idx = static_cast<uint64_t> (gr) * static_cast<uint64_t> (N) + static_cast<uint64_t> (c);
                rowp[c] = make_randn_float(seed, idx); 
            }
        }
    };

    if (threads <= 1) { worker(0); return; }

    std::vector<std::thread> ts;
    ts.reserve(threads);
    for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t);
    for (auto& th : ts) th.join();
}



GeneratedMatrix generate_random_matrix_multi_numa(
        const GenerateParams& params
) {
    const MatrixCfg& matrix = params.matrix;
    const HostMemoryCfg& host = params.host_memory;

    GeneratedMatrix out;
    out.M = matrix.M;
    out.N = matrix.N;

    const int64_t M = matrix.M;
    const int64_t N = matrix.N;

    if (!host.numa_aware || host.placement.empty()) {
        // fallback: single node 0
        NodeFrac nf{0, 1.0};
        std::vector<NodeFrac> placement{nf};
        auto rows = split_rows_by_frac(M, placement);

        NodeBuffer b;
        b.node = 0;
        b.row_start = 0;
        b.rows = M;
        size_t bytes = (size_t)M * (size_t)N * sizeof(float);
        b.bytes = bytes;
        b.ptr = static_cast<float*> (numa_alloc_or_throw(bytes, b.node));


        int threads = (host.max_threads > 0) ? std::min(host.max_threads, host.threads_per_node)
                                             : host.threads_per_node;
        threads = std::max(1, threads);

        fill_rows_f32(b.ptr, b.rows, N, matrix.seed, b.row_start, threads, host.pin_threads, b.node);

        cudaError_t st = cudaHostRegister(b.ptr, b.bytes, cudaHostRegisterDefault);
        if (st == cudaSuccess) {
            b.pinned = true;
        }
        else {
            b.pinned = false;
            cudaGetLastError(); // clear sticky error if you want
            std::cout << "false here" << std::endl;
        }

        out.bufs.push_back(b);
        return out;
    }


    auto rows_per = split_rows_by_frac(M, host.placement);

    int64_t row_cursor = 0;
    out.bufs.reserve(rows_per.size());

    for (size_t i = 0 ; i < rows_per.size() ; i ++ ) {
        NodeBuffer b;

        int node = host.placement[i].node;
        int64_t row = rows_per[i];
        b.row_start = row_cursor;
        b.rows = row;
        b.node = node;

        size_t bytes = static_cast<size_t> (row) * static_cast<size_t> (N) * sizeof(float);
        b.bytes = bytes;

        b.ptr = static_cast<float*> (numa_alloc_or_throw(bytes, b.node));

        out.bufs.push_back(b);

        row_cursor += row;
    }

    std::vector<std::thread> node_threads;
    node_threads.reserve(out.bufs.size());

    for (auto& b : out.bufs) {
        node_threads.emplace_back([&, pb=&b] {
            int threads = host.threads_per_node;
            if (host.max_threads > 0) threads = std::min(threads, host.max_threads);
            threads = std::max(1, threads);

            fill_rows_f32(pb->ptr, pb->rows, N, matrix.seed, pb->row_start,
                      threads, host.pin_threads, pb->node);

            print_numa_residency(pb->ptr, pb->bytes);

            cudaError_t st = cudaHostRegister(pb->ptr, pb->bytes, cudaHostRegisterDefault);
            pb->pinned = (st == cudaSuccess);
            if (st != cudaSuccess) {
                std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(st) << "\n";
            }
        });
    }

    for (auto& th : node_threads) th.join();
    return out;
}






