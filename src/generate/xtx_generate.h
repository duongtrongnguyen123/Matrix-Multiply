#include <cstdint>
#include <cstddef>


struct GenerateParams {
    int64_t M;
    int64_t N;
    uint64_t seed;
    int gpu_local_node;
    int remote_node;
    double split_ratio;
};

void generate_random_matrix_2_numa(const GenerateParams& params, 
                                   float** X_local, int64_t* rows_local,
                                   float** X_remote, int64_t* rows_remote
); 

void free_2_numa(float* X_local, size_t bytes_local, 
                 float* X_remote, size_t bytes_remote);

void alloc_2_numa_parallel(
    size_t bytes_local, int node_local, void** out_local,
    size_t bytes_remote, int node_remote, void** out_remote,
    bool do_first_touch
);
