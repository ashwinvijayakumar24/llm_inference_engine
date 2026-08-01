#!/bin/bash
# Build the custom CUDA decode-attention kernels.
# Run from repo root on PACE after: module load cuda/12.9.1
#
# CUDA_ARCH selects the target compute capability. It was hardcoded to 80 (A100),
# which silently produces a binary that will not run on anything else — and PACE
# Phoenix has far more than A100s available:
#
#     sm_80  A100          gpu-a100        (2/node, one node with 8)
#     sm_89  L40S          gpu-l40s        (8/node, 0.78x the A100 SU rate)
#     sm_90  H100 / H200   gpu-h100/h200   (8/node, 2.43x the A100 SU rate)
#
# Override for a different GPU:
#     CUDA_ARCH=89 bash scripts/build_kernels.sh     # L40S
#     CUDA_ARCH=90 bash scripts/build_kernels.sh     # H100/H200
#
# Multiple architectures in one binary (larger, portable across nodes):
#     CUDA_ARCH="80;89;90" bash scripts/build_kernels.sh
#
# Detect the arch of the GPU you are actually sitting on:
#     nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. "8.0" -> 80
set -e

CUDA_ARCH="${CUDA_ARCH:-80}"
BUILD_DIR="${BUILD_DIR:-build}"

echo "=== Building engine_kernels (CUDA_ARCH=${CUDA_ARCH}) ==="

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Run 'module load cuda/12.9.1' first." >&2
    exit 1
fi

# Warn — do not fail — when the build target does not match the visible GPU.
# Cross-compiling for another node is legitimate; silently shipping a binary
# that cannot launch there is not.
if command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
               | head -1 | tr -d '.' || true)
    if [ -n "$DETECTED" ] && [ "$CUDA_ARCH" != "$DETECTED" ] \
       && [ "${CUDA_ARCH#*"$DETECTED"}" = "$CUDA_ARCH" ]; then
        echo "WARNING: building for sm_${CUDA_ARCH} but this node reports sm_${DETECTED}."
        echo "         Kernels will not launch here. Set CUDA_ARCH=${DETECTED} to fix."
    fi
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ../kernels \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build . -- -j4
echo "=== Build complete: ${BUILD_DIR}/engine_kernels*.so (sm_${CUDA_ARCH}) ==="
