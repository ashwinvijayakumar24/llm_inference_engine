#!/bin/bash
# Run from repo root on PACE after: module load cuda/12.9.1
set -e

echo "=== Building engine_kernels ==="
mkdir -p build
cd build
cmake ../kernels \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . -- -j4
echo "=== Build complete: build/engine_kernels*.so ==="
