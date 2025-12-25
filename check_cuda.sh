#!/bin/bash
# Quick CUDA check script
# Usage: ./check_cuda.sh

echo "=== CUDA Environment Check ==="
echo ""

# Check nvcc
echo "1. Checking nvcc..."
if command -v nvcc &> /dev/null; then
    echo "   ✓ nvcc found: $(which nvcc)"
    nvcc --version | head -n 1
else
    echo "   ✗ nvcc not found in PATH"
    echo "   Searching common locations..."
    
    CUDA_PATHS=(
        "/usr/local/cuda"
        "/usr/local/cuda-11.8"
        "/usr/local/cuda-12.0"
        "/opt/cuda"
        "/opt/cuda-11.8"
    )
    
    found=0
    for path in "${CUDA_PATHS[@]}"; do
        if [ -f "$path/bin/nvcc" ]; then
            echo "   ✓ Found at: $path"
            echo "   → Set: export CUDA_HOME=$path"
            found=1
        fi
    done
    
    if [ $found -eq 0 ]; then
        echo "   ✗ CUDA not found in common locations"
    fi
fi
echo ""

# Check CUDA_HOME
echo "2. Checking CUDA_HOME..."
if [ -n "$CUDA_HOME" ]; then
    echo "   ✓ CUDA_HOME=$CUDA_HOME"
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        echo "   ✓ nvcc exists at CUDA_HOME"
    else
        echo "   ✗ nvcc not found at CUDA_HOME/bin/nvcc"
    fi
else
    echo "   ⚠ CUDA_HOME not set"
fi
echo ""

# Check GPU
echo "3. Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi found"
    echo "   GPU Info:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1
else
    echo "   ⚠ nvidia-smi not found (may not be on GPU node)"
fi
echo ""

# Check compiler
echo "4. Checking compiler..."
if command -v g++ &> /dev/null; then
    echo "   ✓ g++ found: $(which g++)"
    g++ --version | head -n 1
else
    echo "   ✗ g++ not found"
fi
echo ""

# Check CMake
echo "5. Checking CMake..."
if command -v cmake &> /dev/null; then
    echo "   ✓ cmake found: $(which cmake)"
    cmake --version | head -n 1
else
    echo "   ✗ cmake not found"
fi
echo ""

echo "=== Summary ==="
if command -v nvcc &> /dev/null || [ -n "$CUDA_HOME" ]; then
    echo "✓ CUDA appears to be available"
    echo ""
    echo "To build, run:"
    echo "  ./build_cluster_simple.sh"
else
    echo "✗ CUDA setup needs attention"
    echo ""
    echo "Try:"
    echo "  1. Find CUDA: find /usr/local /opt -name nvcc 2>/dev/null"
    echo "  2. Set CUDA_HOME: export CUDA_HOME=/path/to/cuda"
    echo "  3. Add to PATH: export PATH=\$CUDA_HOME/bin:\$PATH"
fi

