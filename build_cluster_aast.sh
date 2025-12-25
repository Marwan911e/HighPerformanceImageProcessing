#!/bin/bash
# Build script for AAST cluster
# Uses submit.nvcc for CUDA compilation
# Usage: ./build_cluster_aast.sh

set -e  # Exit on error

echo "=== CUDA Image Processor - AAST Cluster Build ==="
echo ""

# Check if submit.nvcc exists
if ! command -v submit.nvcc &> /dev/null; then
    echo "ERROR: submit.nvcc not found"
    echo "This script is designed for AAST cluster with submit.nvcc wrapper"
    exit 1
fi

echo "Using cluster's submit.nvcc compiler"
CUDA_VER_OUT=$(submit.nvcc --version 2>&1)
echo "$CUDA_VER_OUT"
echo ""

# Check CUDA version and set appropriate C++ standard
# CUDA 10.0 and earlier don't support C++17, use C++14
if echo "$CUDA_VER_OUT" | grep -q "release 10\."; then
    CPP_STD="c++14"
    echo "Detected CUDA 10.x - Using C++14 (C++17 not supported)"
else
    CPP_STD="c++17"
    echo "Detected CUDA 11.x or later - Using C++17"
fi

# Find CUDA include directory for g++ compilation
# Priority: 1) extract from submit.nvcc (ALWAYS first, ignore Python packages), 2) CUDA_HOME/CUDA_PATH (if not Python), 3) common locations
CUDA_INC_DIR=""

# Check if CUDA_HOME is set to a Python package (should be ignored)
if [ -n "$CUDA_HOME" ]; then
    if echo "$CUDA_HOME" | grep -q "site-packages\|python\|llamaenv"; then
        echo "WARNING: CUDA_HOME points to Python package, ignoring it: $CUDA_HOME"
        unset CUDA_HOME
    fi
fi

# First, try direct check of /usr/local/cuda-10.0 (we know from submit.nvcc it's there)
if [ -d "/usr/local/cuda-10.0/include" ] && [ -f "/usr/local/cuda-10.0/include/cuda_runtime.h" ]; then
    CUDA_INC_DIR="/usr/local/cuda-10.0/include"
    echo "Found CUDA at standard location: $CUDA_INC_DIR"
fi

# If not found, try to extract from submit.nvcc (most reliable, ignores environment)
if [ -z "$CUDA_INC_DIR" ]; then
    echo "Extracting CUDA path from submit.nvcc..."
    NVCC_VERBOSE=$(submit.nvcc -E -x cu - -v < /dev/null 2>&1)
    CUDA_INC_LINE=$(echo "$NVCC_VERBOSE" | grep "INCLUDES=" | head -1)
    
    if [ -n "$CUDA_INC_LINE" ]; then
        # Extract path after -I (handles both quoted and unquoted)
        # Pattern: -I/usr/local/cuda-10.0/bin/..//include or -I"/usr/local/cuda-10.0/bin/..//include"
        FULL_PATH=$(echo "$CUDA_INC_LINE" | sed 's/.*-I"\([^"]*\)".*/\1/')
        if [ "$FULL_PATH" = "$CUDA_INC_LINE" ]; then
            # Try without quotes - match -I followed by path
            FULL_PATH=$(echo "$CUDA_INC_LINE" | sed 's/.*-I\([^ "]*\).*/\1/')
        fi
        
        if [ -n "$FULL_PATH" ] && [ "$FULL_PATH" != "$CUDA_INC_LINE" ]; then
            # Extract base directory: everything before /bin
            CUDA_BASE=$(echo "$FULL_PATH" | sed 's|/bin/.*||')
            CUDA_BASE=$(echo "$CUDA_BASE" | sed 's|//|/|g')
            
            # Check if base/include exists and is NOT a Python package
            if [ -n "$CUDA_BASE" ] && [ -d "$CUDA_BASE/include" ] && [ -f "$CUDA_BASE/include/cuda_runtime.h" ]; then
                if ! echo "$CUDA_BASE" | grep -q "site-packages\|python\|llamaenv"; then
                    CUDA_INC_DIR="$CUDA_BASE/include"
                    echo "Found CUDA from submit.nvcc: $CUDA_INC_DIR"
                fi
            fi
        fi
    fi
fi

# If not found from submit.nvcc, try environment variables (but skip Python packages)
if [ -z "$CUDA_INC_DIR" ]; then
    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/include" ] && [ -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
        if ! echo "$CUDA_HOME" | grep -q "site-packages\|python\|llamaenv"; then
            CUDA_INC_DIR="$CUDA_HOME/include"
            echo "Found CUDA from CUDA_HOME: $CUDA_INC_DIR"
        fi
    elif [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH/include" ] && [ -f "$CUDA_PATH/include/cuda_runtime.h" ]; then
        if ! echo "$CUDA_PATH" | grep -q "site-packages\|python\|llamaenv"; then
            CUDA_INC_DIR="$CUDA_PATH/include"
            echo "Found CUDA from CUDA_PATH: $CUDA_INC_DIR"
        fi
    fi
fi

# If still not found, try common locations
if [ -z "$CUDA_INC_DIR" ]; then
    for path in /usr/local/cuda-10.0/include /usr/local/cuda/include /opt/cuda/include; do
        if [ -d "$path" ] && [ -f "$path/cuda_runtime.h" ]; then
            CUDA_INC_DIR="$path"
            echo "Found CUDA in common location: $CUDA_INC_DIR"
            break
        fi
    done
fi

# Verify we found a valid CUDA include directory
if [ -z "$CUDA_INC_DIR" ] || [ ! -d "$CUDA_INC_DIR" ] || [ ! -f "$CUDA_INC_DIR/cuda_runtime.h" ]; then
    echo ""
    echo "ERROR: Cannot find valid CUDA include directory!"
    echo ""
    echo "From submit.nvcc, CUDA should be at: /usr/local/cuda-10.0"
    echo "Try setting manually:"
    echo "  export CUDA_HOME=/usr/local/cuda-10.0"
    echo "  ./build_cluster_aast.sh"
    echo ""
    echo "Or run: ./find_cuda.sh  (if available)"
    exit 1
fi

CUDA_INC_FLAG="-I$CUDA_INC_DIR"
echo "Using CUDA includes: $CUDA_INC_DIR"
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Compile CUDA kernels
echo "Compiling CUDA kernels..."
submit.nvcc -c ../src/cuda_kernels.cu -o cuda_kernels.o -I../include -I../lib -arch=sm_70 -O3 -std=$CPP_STD

# Compile C++ sources
echo "Compiling C++ sources..."
g++ -c ../src/main_cuda.cpp -o main_cuda.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/image.cpp -o image.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/cuda_utils.cpp -o cuda_utils.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/filters_cuda.cpp -o filters_cuda.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/edge_detection_cuda.cpp -o edge_detection_cuda.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/point_operations.cpp -o point_operations.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/noise.cpp -o noise.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/morphological.cpp -o morphological.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/geometric.cpp -o geometric.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC
g++ -c ../src/color_operations.cpp -o color_operations.o -I../include -I../lib $CUDA_INC_FLAG -O3 -std=$CPP_STD -fPIC

# Link
echo "Linking executable..."
g++ main_cuda.o image.o cuda_utils.o filters_cuda.o edge_detection_cuda.o \
    point_operations.o noise.o morphological.o geometric.o color_operations.o \
    cuda_kernels.o -o image_processor_cuda -lcudart -lm

cd ..

echo ""
echo "=== Build Complete ==="
echo "Executable: build/image_processor_cuda"
echo ""
echo "To submit job:"
echo "  sbatch submit.gpu \"./build/image_processor_cuda\""
echo ""
echo "To monitor jobs:"
echo "  watch -n 1 \"squeue\""

