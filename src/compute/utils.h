#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdexcept>
#include <string>
#include <fstream>


inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("[cuda] ") + msg + ": " + cudaGetErrorString(e));
    }
}

inline void cublas_check(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("[cublas] ") + msg +
                                 " (status=" + std::to_string((int)s) + ")");
    }
}

inline void cublaslt_check(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("[cublasLt] ") + msg +
                                 " (status=" + std::to_string((int)s) + ")");
    }
}


inline std::string lower(std::string x) {
    for (char& c : x) c = (char)std::tolower((unsigned char)c);
    return x;
}

inline cublasMath_t parse_math_mode(std::string s) {
    if (s.empty()) return CUBLAS_DEFAULT_MATH;
    s = lower(std::move(s));

    if (s == "default")  return CUBLAS_DEFAULT_MATH;
    if (s == "tf32")     return CUBLAS_TF32_TENSOR_OP_MATH;
    if (s == "pedantic") return CUBLAS_PEDANTIC_MATH;

    // common aliases (treat as default math unless you want different behavior)
    if (s == "tensor_op" || s == "tensorop" || s == "tensor") return CUBLAS_DEFAULT_MATH;

#if defined(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
    if (s == "disallow_reduced_precision_reduction") {
        return CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
    }
#endif

    return CUBLAS_DEFAULT_MATH;
}



inline cublasFillMode_t parse_triangle(const std::string& tri) {
    if (tri == "upper") return CUBLAS_FILL_MODE_UPPER;
    return CUBLAS_FILL_MODE_LOWER;
}

inline int read_int_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return -1;
    int x = -1;
    f >> x;
    return x;
}

inline int gpu_to_numa_node(int device_id) {
    char pci[32];
    cudaDeviceGetPCIBusId(pci, sizeof(pci), device_id);
    // pci looks like "0000:65:00.0"

    std::string path = std::string("/sys/bus/pci/devices/") + pci + "/numa_node";
    int node = read_int_file(path);

    if (node < 0) node = 0; // fallback
    return node;
}