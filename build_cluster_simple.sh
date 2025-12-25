#!/bin/bash
# Simple build script for cluster without module system
# Usage: ./build_cluster_simple.sh

set -e  # Exit on error

echo "=== CUDA Image Processor - Simple Cluster Build ==="
echo ""

# Check for CUDA in common locations
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-11.8"
    "/usr/local/cuda-12.0"
    "/opt/cuda"
    "/opt/cuda-11.8"
    "$HOME/cuda"
)

CUDA_HOME=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
        CUDA_HOME="$path"
        echo "Found CUDA at: $CUDA_HOME"
        break
    fi
done

if [ -z "$CUDA_HOME" ]; then
    # Try to find nvcc in PATH
    if command -v nvcc &> /dev/null; then
        CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        echo "Found CUDA via nvcc in PATH: $CUDA_HOME"
    else
        echo "ERROR: CUDA not found. Please:"
        echo "  1. Set CUDA_HOME environment variable: export CUDA_HOME=/path/to/cuda"
        echo "  2. Or ensure nvcc is in your PATH"
        exit 1
    fi
fi

# Check CUDA
echo "Checking CUDA installation..."
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    $CUDA_HOME/bin/nvcc --version | head -n 1
else
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

# Set environment
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check compiler
echo ""
echo "Checking compiler..."
if command -v g++ &> /dev/null; then
    g++ --version | head -n 1
else
    echo "ERROR: g++ not found"
    exit 1
fi

# Check CMake
echo ""
echo "Checking CMake..."
if command -v cmake &> /dev/null; then
    cmake --version | head -n 1
else
    echo "ERROR: cmake not found"
    echo "You may need to install cmake or use a different build method"
    exit 1
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Configure CMake
echo ""
echo "Configuring CMake..."
cmake .. -DENABLE_CUDA=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
         -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
         -DCMAKE_CXX_COMPILER=g++

# Build
echo ""
echo "Building application..."
make -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "=== Build Complete ==="
echo "Executable: build/image_processor_cuda"
echo ""
echo "To run:"
echo "  ./build/image_processor_cuda"

