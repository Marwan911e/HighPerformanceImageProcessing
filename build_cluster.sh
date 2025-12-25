#!/bin/bash
# Build script for HPC cluster
# Usage: ./build_cluster.sh

set -e  # Exit on error

echo "=== CUDA Image Processor - Cluster Build Script ==="

# Check if modules are loaded
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please load CUDA module:"
    echo "  module load cuda/11.8"
    exit 1
fi

# Check CUDA
echo "Checking CUDA installation..."
nvcc --version
echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure CMake
echo "Configuring CMake..."
cmake .. -DENABLE_CUDA=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_COMPILER=nvcc \
         -DCMAKE_CXX_COMPILER=g++

# Build
echo "Building application..."
make -j$(nproc) || make -j4

echo ""
echo "=== Build Complete ==="
echo "Executable: build/image_processor_cuda"
echo ""
echo "To run interactively:"
echo "  srun --gres=gpu:1 --pty bash"
echo "  ./build/image_processor_cuda"
echo ""
echo "To submit as job:"
echo "  sbatch submit_cuda.sh"

