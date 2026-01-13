FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libnuma-dev \
    libyaml-cpp-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY build.sh .

# Build
RUN nvcc -std=c++17 -O3 -Xcompiler -fopenmp -I./src \
    src/main.cu \
    src/config/xtx_config.cpp \
    src/generate/xtx_generate.cu \
    src/compute/xtx_compute.cu \
    src/compute/xtx_cublas.cu \
    src/compute/xtx_cublasLt.cu \
    -lcublasLt -lcublas -lcudart -lnuma -lyaml-cpp -lgomp \
    -o xtx_bench

# Default command
CMD ["./xtx_bench", "configs/xtx_precision_perf.yaml"]
