#include "io/npy_save.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <fstream>
#include <ios>
#include <string>


bool save_npy_fp32(const std::string& path, const float* data_row_major, std::int64_t N) {
    if (!data_row_major || N <= 0) return false;

    // .npy v1.0 header (little-endian float32, C-order)
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                         std::to_string(N) + ", " + std::to_string(N) + "), }";

    // Pad to 16-byte alignment with spaces and newline at end
    size_t header_len = header.size() + 1; // + newline
    size_t preamble = 10; // magic(6) + ver(2) + headerlen(2)
    size_t pad = 16 - ((preamble + header_len) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');

    uint16_t hlen =  static_cast<uint16_t> (header.size());

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    f.write((const char*)magic, 6);
    unsigned char ver[2] = {1, 0};
    f.write((const char*)ver, 2);
    f.write((const char*)&hlen, 2);
    f.write(header.data(), (std::streamsize)header.size());

    const size_t bytes = (size_t)N * (size_t)N * sizeof(float);
    f.write((const char*)data_row_major, (std::streamsize)bytes);

    return (bool)f;

}