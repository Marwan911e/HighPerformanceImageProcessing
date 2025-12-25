#!/bin/bash
# Check cluster setup for CUDA build
# Usage: ./check_cluster_setup.sh

echo "=== CUDA Cluster Setup Checker ==="
echo ""

# Check CUDA
echo "1. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "   ✓ nvcc found"
    nvcc --version | head -n 1
else
    echo "   ✗ nvcc not found"
    echo "   → Try: module load cuda/11.8"
fi
echo ""

# Check CUDA environment
echo "2. Checking CUDA environment variables..."
if [ -n "$CUDA_HOME" ]; then
    echo "   ✓ CUDA_HOME=$CUDA_HOME"
else
    echo "   ⚠ CUDA_HOME not set"
fi

if [ -n "$CUDA_PATH" ]; then
    echo "   ✓ CUDA_PATH=$CUDA_PATH"
else
    echo "   ⚠ CUDA_PATH not set"
fi
echo ""

# Check GPU
echo "3. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1
else
    echo "   ⚠ nvidia-smi not found (may not be on GPU node)"
fi
echo ""

# Check compiler
echo "4. Checking compiler..."
if command -v g++ &> /dev/null; then
    echo "   ✓ g++ found"
    g++ --version | head -n 1
else
    echo "   ✗ g++ not found"
    echo "   → Try: module load gcc/9.4.0"
fi
echo ""

# Check CMake
echo "5. Checking CMake..."
if command -v cmake &> /dev/null; then
    echo "   ✓ cmake found"
    cmake --version | head -n 1
else
    echo "   ✗ cmake not found"
    echo "   → Try: module load cmake/3.20"
fi
echo ""

# Check modules
echo "6. Checking loaded modules..."
if command -v module &> /dev/null; then
    echo "   Loaded modules:"
    module list 2>/dev/null || echo "   (module command not available)"
else
    echo "   ⚠ module command not found"
fi
echo ""

echo "=== Setup Check Complete ==="
echo ""
echo "If all checks pass, you can build with:"
echo "  ./build_cluster.sh"
echo ""
echo "Or manually:"
echo "  mkdir build && cd build"
echo "  cmake .. -DENABLE_CUDA=ON"
echo "  make -j4"

