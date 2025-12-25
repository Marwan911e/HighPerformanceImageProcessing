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
g++ -c ../src/main_cuda.cpp -o main_cuda.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/image.cpp -o image.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/cuda_utils.cpp -o cuda_utils.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/filters_cuda.cpp -o filters_cuda.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/edge_detection_cuda.cpp -o edge_detection_cuda.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/point_operations.cpp -o point_operations.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/noise.cpp -o noise.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/morphological.cpp -o morphological.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/geometric.cpp -o geometric.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC
g++ -c ../src/color_operations.cpp -o color_operations.o -I../include -I../lib -O3 -std=$CPP_STD -fPIC

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

