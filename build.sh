#!/bin/bash
set -euo pipefail

# ---- Arch: fat (86+89) by default ----
CUDA_ARCH="${CUDA_ARCH:-fat}"
if [ "$CUDA_ARCH" = "fat" ]; then
  GENFLAGS=(
    -gencode arch=compute_86,code=sm_86
    -gencode arch=compute_89,code=sm_89
  )
elif [ "$CUDA_ARCH" = "86" ] || [ "$CUDA_ARCH" = "89" ]; then
  GENFLAGS=(-arch=sm_${CUDA_ARCH})
else
  echo "[error] CUDA_ARCH must be: fat | 86 | 89 (got: $CUDA_ARCH)"
  exit 1
fi

# ---- nvcc ----
if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
  NVCC="${CUDA_HOME}/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
else
  echo "[error] nvcc not found. Set CUDA_HOME or add nvcc to PATH."
  exit 1
fi

# ---- host compiler (conda g++ first, then system) ----
CXX=""
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]; then
  CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
elif command -v g++ >/dev/null 2>&1; then
  CXX="$(command -v g++)"
else
  echo "[error] No host C++ compiler found (need g++)."
  echo "Hint: conda install -c conda-forge gxx_linux-64"
  exit 1
fi

# ---- NVTX (optional) ----
NVTX_INC=""
NVTX_REAL_LIB=""
NVTX_SHIM_LIB=""

if [ -n "${CONDA_PREFIX:-}" ]; then
  NVTX_INC="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/nvtx/include"
  NVTX_REAL_LIB="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/nvtx/lib"
  NVTX_SHIM_LIB="${HOME}/nvtxlib"

  if [ -d "$NVTX_INC" ] && [ -d "$NVTX_REAL_LIB" ]; then
    mkdir -p "$NVTX_SHIM_LIB"
    ln -sf "$NVTX_REAL_LIB/libnvToolsExt.so.1" "$NVTX_SHIM_LIB/libnvToolsExt.so"
  else
    NVTX_INC=""
    NVTX_REAL_LIB=""
    NVTX_SHIM_LIB=""
  fi
fi

echo "[build] NVCC: $NVCC"
"$NVCC" --version | head -n 5 | sed 's/^/[build] /'
echo "[build] CXX : $CXX"
"$CXX" --version | head -n 1 | sed 's/^/[build] /'
echo "[build] CUDA_ARCH: $CUDA_ARCH"

# ---- include paths ----
INCLUDES=(-I./src)
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/include" ]; then
  INCLUDES+=(-I"${CONDA_PREFIX}/include")
fi
if [ -n "$NVTX_INC" ]; then
  INCLUDES+=(-I"$NVTX_INC")
fi

# ---- lib paths + rpath ----
LIBDIRS=()
RPATH_FLAGS=()

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  LIBDIRS+=(-L"${CONDA_PREFIX}/lib")
  RPATH_FLAGS+=(-Xlinker -rpath -Xlinker "${CONDA_PREFIX}/lib")
fi
if [ -n "$NVTX_SHIM_LIB" ]; then
  LIBDIRS+=(-L"$NVTX_SHIM_LIB")
  RPATH_FLAGS+=(-Xlinker -rpath -Xlinker "$NVTX_SHIM_LIB")
fi
if [ -n "$NVTX_REAL_LIB" ]; then
  RPATH_FLAGS+=(-Xlinker -rpath -Xlinker "$NVTX_REAL_LIB")
fi

SRCS=(
  src/main.cu
  src/config/xtx_config.cpp
  src/generate/xtx_generate.cu
  src/compute/xtx_compute.cu
  src/compute/xtx_cublas.cu
  src/compute/xtx_cublasLt.cu
)

set -x
"$NVCC" -std=c++17 -O3 -Xcompiler -fopenmp \
  -ccbin "$CXX" \
  "${GENFLAGS[@]}" \
  -lineinfo \
  "${INCLUDES[@]}" \
  "${SRCS[@]}" \
  "${LIBDIRS[@]}" \
  "${RPATH_FLAGS[@]}" \
  -lcublasLt -lcublas -lcudart \
  -Xlinker --start-group \
    -lnuma -lyaml-cpp -lnvToolsExt \
  -Xlinker --end-group \
  -lgomp \
  -o xtx_bench
set +x

echo "[build] done -> ./xtx_bench"

