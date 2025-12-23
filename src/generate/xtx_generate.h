#include <config/xtx_config.h>

#include <cstdint>
#include <cstddef>

void generate_random_matrix_2_numa(const Config& cfg, 
                                   float** X_local, int64_t* rows_local,
                                   float** X_remote, int64_t* rows_remote
); 

void free_2_numa(float* X_local, size_t bytes_local, 
                 float* X_remote, size_t bytes_remote);

