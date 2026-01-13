#!/bin/bash
set -e

# ---------- choose nvcc ----------
if [ -x /usr/local/cuda/bin/nvcc ]; then
  NVCC=/usr/local/cuda/bin/nvcc
  echo "[build] using system nvcc"
elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
  NVCC="$CONDA_PREFIX/bin/nvcc"
  echo "[build] using conda nvcc"
else
  echo "[error] nvcc not found (need system CUDA toolkit or conda nvcc)"
  exit 1
fi

# ---------- compiler ----------
CXX="${CXX:-$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++}"

# ---------- NVTX ----------
NVTX_INC="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/include"
NVTX_REAL_LIB="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvtx/lib"
NVTX_SHIM_LIB="$HOME/nvtxlib"
mkdir -p "$NVTX_SHIM_LIB"
ln -sf "$NVTX_REAL_LIB/libnvToolsExt.so.1" "$NVTX_SHIM_LIB/libnvToolsExt.so"

# ---------- arch flags ----------
GENFLAGS="
  -gencode arch=compute_86,code=sm_86
  -gencode arch=compute_89,code=sm_89
"

# ---------- build ----------
$NVCC -std=c++17 -O3 -Xcompiler -fopenmp \
  -ccbin "$CXX" \
  $GENFLAGS \
  -I./src \
  -I"$CONDA_PREFIX/include" \
  -I"$NVTX_INC" \
  src/main.cu \
  src/config/xtx_config.cpp \
  src/generate/xtx_generate.cu \
  src/compute/xtx_compute.cu \
  src/compute/xtx_cublas.cu \
  src/compute/xtx_cublasLt.cu \
  src/io/npy_save.cu \
  -L"$CONDA_PREFIX/lib" \
  -L"$NVTX_SHIM_LIB" \
  -Xlinker -rpath -Xlinker "$CONDA_PREFIX/lib" \
  -Xlinker -rpath -Xlinker "$NVTX_REAL_LIB" \
  -Xlinker -rpath -Xlinker "$NVTX_SHIM_LIB" \
  -lcublasLt -lcublas -lcudart \
  -Xlinker --start-group \
    -lnuma -lyaml-cpp -lnvToolsExt \
  -Xlinker --end-group \
  -lgomp \
  -o xtx_bench

