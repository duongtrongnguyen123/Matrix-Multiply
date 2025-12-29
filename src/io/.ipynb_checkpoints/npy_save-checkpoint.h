#pragma once
#include <cstdint>
#include <string>

bool save_npy_fp32(
    const std::string& path,
    const float* data_row_major,
    std::int64_t N
);