#!/bin/bash
# Build inside WSL. No system CUDA toolkit here: nvcc 13.0 and cuBLAS 13 come
# from pip wheels, the host compiler from conda-forge (the system gcc has no
# C++ front end).
set -euo pipefail
cd "$(dirname "$0")"
CUDA=${CUDA_HOME:-$HOME/inferscope/cuda130/nvidia/cu13}
LIBS=${CUBLAS_HOME:-$HOME/inferscope/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13}
CXX=${CXX:-$(ls $HOME/mamba/envs/*/bin/x86_64-conda-linux-gnu-g++ | head -1)}
mkdir -p build/include
# Only cuBLAS's headers from the wheel: its include dir also carries CUDA 13.4
# runtime headers that would shadow nvcc 13.0's own and break the build.
for h in "$LIBS"/include/cublas*.h; do ln -sf "$h" build/include/; done
"$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_86 -lineinfo -ccbin "$CXX" \
  -Ibuild/include src/bench.cu -o build/bench \
  -L"$LIBS/lib" -l:libcublas.so.13 -l:libcublasLt.so.13 -Xlinker -rpath,"$LIBS/lib"
echo "built build/bench"
