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
CUDA_INC_DIR=""
if [ -n "$CUDA_HOME" ]; then
    CUDA_INC_DIR="$CUDA_HOME/include"
elif [ -n "$CUDA_PATH" ]; then
    CUDA_INC_DIR="$CUDA_PATH/include"
else
    # Try common locations
    for path in /usr/local/cuda/include /usr/local/cuda-10.0/include /opt/cuda/include /usr/include; do
        if [ -d "$path" ] && [ -f "$path/cuda_runtime.h" ]; then
            CUDA_INC_DIR="$path"
            break
        fi
    done
    
    # If still not found, search for cuda_runtime.h
    if [ -z "$CUDA_INC_DIR" ]; then
        CUDA_INC_DIR=$(find /usr /opt -name "cuda_runtime.h" 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
    fi
fi

if [ -z "$CUDA_INC_DIR" ] || [ ! -d "$CUDA_INC_DIR" ]; then
    echo "WARNING: Cannot find CUDA include directory automatically."
    echo "Attempting to extract from submit.nvcc..."
    
    # Try to get include path from submit.nvcc by checking its verbose output
    NVCC_VERBOSE=$(submit.nvcc -E -x cu - -v < /dev/null 2>&1)
    
    # Extract include path from verbose output
    # Line looks like: #$ INCLUDES="-I/usr/local/cuda-10.0/bin/..//include"
    CUDA_INC_LINE=$(echo "$NVCC_VERBOSE" | grep "INCLUDES=" | head -1)
    
    if [ -n "$CUDA_INC_LINE" ]; then
        # Extract the path: match -I"path" and extract path
        # Pattern: -I/usr/local/cuda-10.0/bin/..//include
        CUDA_INC_FROM_NVCC=$(echo "$CUDA_INC_LINE" | sed 's/.*-I"\([^"]*\)".*/\1/' | sed 's/.*-I\([^ ]*\).*/\1/')
        
        # Clean up the path (remove /bin/..// and normalize)
        if [ -n "$CUDA_INC_FROM_NVCC" ]; then
            # Remove /bin/..// and normalize double slashes
            CUDA_INC_FROM_NVCC=$(echo "$CUDA_INC_FROM_NVCC" | sed 's|/bin/\.\.//|/|g' | sed 's|//|/|g')
            
            # Verify the path exists and contains cuda_runtime.h
            if [ -d "$CUDA_INC_FROM_NVCC" ] && [ -f "$CUDA_INC_FROM_NVCC/cuda_runtime.h" ]; then
                CUDA_INC_DIR="$CUDA_INC_FROM_NVCC"
            fi
        fi
        
        # Alternative: extract CUDA base directory (e.g., /usr/local/cuda-10.0)
        if [ -z "$CUDA_INC_DIR" ] || [ ! -d "$CUDA_INC_DIR" ]; then
            # Extract base path before /bin/..//include
            CUDA_BASE=$(echo "$CUDA_INC_LINE" | sed 's|.*-I"\([^"]*\)/bin/\.\.//include".*|\1|' | sed 's|.*-I\([^ ]*\)/bin/\.\.//include.*|\1|')
            if [ -n "$CUDA_BASE" ] && [ "$CUDA_BASE" != "$CUDA_INC_LINE" ]; then
                CUDA_BASE=$(echo "$CUDA_BASE" | sed 's|//|/|g')
                if [ -d "$CUDA_BASE/include" ] && [ -f "$CUDA_BASE/include/cuda_runtime.h" ]; then
                    CUDA_INC_DIR="$CUDA_BASE/include"
                fi
            fi
        fi
    fi
    
    if [ -n "$CUDA_INC_DIR" ] && [ -d "$CUDA_INC_DIR" ]; then
        echo "Found CUDA includes from submit.nvcc: $CUDA_INC_DIR"
    else
        echo ""
        echo "ERROR: Cannot find CUDA include directory!"
        echo ""
        echo "From submit.nvcc verbose output, CUDA appears to be at: /usr/local/cuda-10.0"
        echo "Try setting:"
        echo "  export CUDA_HOME=/usr/local/cuda-10.0"
        echo ""
        echo "Or run: ./find_cuda.sh  (if available)"
        echo "Then run this script again."
        exit 1
    fi
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

