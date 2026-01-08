#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <numa.h>
#include <config/xtx_config.h>

struct NodeBuffer {
    int node;
    int64_t row_start;   // global row offset
    int64_t rows;
    size_t bytes;
    float* ptr;
    bool pinned = true;
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
    const MatrixCfg& matrix;
    const HostMemoryCfg& host_memory;
};

void pin_thread_to_numa_node(int node);

GeneratedMatrix generate_random_matrix_multi_numa(
        const GenerateParams& params );


