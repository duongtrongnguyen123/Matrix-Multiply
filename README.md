# System-Level Optimization of Large-Scale Matrix Multiplication on GPUs

Report (PDF): [out-of-core-gpu-matmul-report.pdf](docs/out-of-core-gpu-matmul-report.pdf)

Matrix multiplication is a core operation in large-scale data analysis and machine learning workloads. When matrix sizes exceed device memory capacity, overall performance is often dominated not by compute throughput but by host-device data movement and memory management.

This project focuses on engineering efficient end-to-end matrix multiplication pipelines on NVIDIA GPU systems using cuBLAS and cuBLASLt as the compute backend. Rather than implementing custom GPU kernels, the work explores system-level optimizations including:

- Pinned host memory allocation
- NUMA-aware memory placement
- Multi-threaded data staging
- Asynchronous transfers
- Double buffering
- Multi-GPU execution

## Requirements

- CUDA Toolkit (tested with CUDA 12.x)
- cuBLAS / cuBLASLt
- libnuma
- yaml-cpp
- OpenMP
- C++17 compiler

## Build

Recommended (compiles for the local GPU toolchain and avoids PTX mismatch issues on other devices):
```bash
./build.sh
```

Direct nvcc build (may fail on a different GPU with `the provided PTX was compiled with an unsupported toolchain`):
```bash
nvcc -std=c++17 -O3 -Xcompiler -fopenmp -I./src \
  src/main.cu \
  src/config/xtx_config.cpp \
  src/generate/xtx_generate.cu \
  src/compute/xtx_compute.cu \
  src/compute/xtx_cublas.cu \
  src/compute/xtx_cublasLt.cu \
  -lcublasLt -lcublas -lcudart -lnuma -lyaml-cpp -lgomp \
  -o xtx_bench
```

## Docker

Build the container:
```bash
docker build -t xtx-bench .
```

Run (requires NVIDIA GPU and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):
```bash
docker run --gpus all xtx-bench
```

Run with custom config:
```bash
docker run --gpus all -v $(pwd)/configs:/app/configs xtx-bench ./xtx_bench configs/your_config.yaml
```

## Usage

```bash
./xtx_bench configs/xtx_precision_perf.yaml
```

## Configuration

See `configs/xtx_precision_perf.yaml` for available options:

- Matrix dimensions (M, N)
- Precision modes: fp32, tf32, bf16, fp16, fp64
- Chunking and buffer settings
- NUMA placement
- Multi-GPU configuration

## Project Structure

```
.
├── src/
│   ├── main.cu              # Entry point
│   ├── config/              # YAML config parser
│   ├── compute/             # cuBLAS compute kernels
│   └── generate/            # Matrix generation
├── configs/                 # YAML configuration files
├── scripts/                 # Evaluation scripts
└── README.md
```
