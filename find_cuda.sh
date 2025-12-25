#!/bin/bash
# Helper script to find CUDA installation on cluster
# Usage: ./find_cuda.sh

echo "=== Finding CUDA Installation ==="
echo ""

# Check environment variables
echo "1. Checking environment variables:"
if [ -n "$CUDA_HOME" ]; then
    echo "   ✓ CUDA_HOME=$CUDA_HOME"
    if [ -d "$CUDA_HOME/include" ]; then
        echo "   ✓ Includes found at: $CUDA_HOME/include"
    fi
fi

if [ -n "$CUDA_PATH" ]; then
    echo "   ✓ CUDA_PATH=$CUDA_PATH"
    if [ -d "$CUDA_PATH/include" ]; then
        echo "   ✓ Includes found at: $CUDA_PATH/include"
    fi
fi
echo ""

# Check submit.nvcc
echo "2. Checking submit.nvcc:"
if command -v submit.nvcc &> /dev/null; then
    echo "   ✓ submit.nvcc found: $(which submit.nvcc)"
    # Try to see what it uses
    submit.nvcc --version 2>&1 | head -3
fi
echo ""

# Check common locations
echo "3. Checking common CUDA locations:"
for path in /usr/local/cuda /usr/local/cuda-10.0 /usr/local/cuda-10 /opt/cuda /opt/cuda-10.0; do
    if [ -d "$path/include" ] && [ -f "$path/include/cuda_runtime.h" ]; then
        echo "   ✓ Found: $path"
        echo "     Includes: $path/include"
    fi
done
echo ""

# Search for cuda_runtime.h
echo "4. Searching for cuda_runtime.h:"
FOUND=$(find /usr /opt -name "cuda_runtime.h" 2>/dev/null | head -3)
if [ -n "$FOUND" ]; then
    echo "   Found at:"
    echo "$FOUND" | while read file; do
        dir=$(dirname "$file")
        echo "     $dir"
    done
else
    echo "   ✗ Not found in /usr or /opt"
fi
echo ""

# Check what submit.nvcc actually uses
echo "5. Checking what submit.nvcc uses:"
if command -v submit.nvcc &> /dev/null; then
    # Try to compile a dummy file to see include paths
    echo "   Attempting to extract include paths..."
    submit.nvcc -E -x cu - -v < /dev/null 2>&1 | grep -A 20 "include" | head -10 || echo "   (Could not extract include paths)"
fi
echo ""

echo "=== Recommendation ==="
if [ -n "$FOUND" ]; then
    CUDA_INC=$(echo "$FOUND" | head -1 | xargs dirname)
    echo "Set CUDA_HOME to:"
    CUDA_BASE=$(dirname "$CUDA_INC")
    echo "  export CUDA_HOME=$CUDA_BASE"
    echo ""
    echo "Or add to build command:"
    echo "  CUDA_HOME=$CUDA_BASE ./build_cluster_aast.sh"
else
    echo "CUDA not found automatically. Please:"
    echo "  1. Ask cluster administrator for CUDA location"
    echo "  2. Or check cluster documentation"
    echo "  3. Or try: find /usr /opt -name 'nvcc' 2>/dev/null"
fi

