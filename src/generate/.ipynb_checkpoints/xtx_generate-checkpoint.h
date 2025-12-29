#pragma once
#include <cstdint>
#include <cstddef>
#include <numa.h> 

struct NodeBuffer {
    int node;
    int64_t row_start;   // global row offset
    int64_t rows; 
    size_t bytes;
    float* ptr;
};

struct GeneratedMatrix{
    int64_t M;
    int64_t N;
    std::vector<NodeBuffer> bufs;
    ~GeneratedMatrix() {
        for (auto& b : bufs) {
            if (b.ptr) {
                numa_free(b.ptr, b.bytes);
                b.ptr = nullptr;
            }
        }
    }
};

struct GenerateParams {
    int64_t M;
    int64_t N;
    uint64_t seed;
    std::vector<NodeFrac> placement;
    int threads_per_node = 8;
    int max_threads = 0;         
    bool pin_threads = true;
    bool numa_aware = true;
};


static std::vector<int64_t> split_rows_by_frac(int64_t M, const std::vector<NodeFrac>& placement);

static void fill_rows_f32(float* base, int64_t rows, int64_t N,                                  uint64_t seed, int64_t global_row_start,
                          int threads, bool pin_threads, int node);

GeneratedMatrix generate_random_matrix_multi_numa(
        const GenerateParams& params );


