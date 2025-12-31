#!/bin/bash

nvcc -std=c++17 -O3 -Xcompiler -fopenmp -I./src \
  src/main.cu \
  src/config/xtx_config.cpp \
  src/generate/xtx_generate.cu \
  src/compute/xtx_compute.cu \
  src/compute/xtx_cublas.cu \
  src/compute/xtx_cublasLt.cu \
  src/io/npy_save.cu \
  -lcublasLt -lcublas -lcudart -lnuma -lyaml-cpp -lgomp \
  -o xtx_bench
