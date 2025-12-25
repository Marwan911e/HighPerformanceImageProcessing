#!/bin/bash
#SBATCH --job-name=image_process_cuda
#SBATCH --output=output_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --partition=gpu            # GPU partition (adjust to your cluster)
#SBATCH --mem=8GB

# Load modules (adjust module names for your cluster)
module load cuda/11.8
module load gcc/9.4.0
module load cmake/3.20

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Build if not already built
if [ ! -f "build/image_processor_cuda" ]; then
    echo "Building CUDA application..."
    mkdir -p build
    cd build
    cmake .. -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    make -j4
    cd ..
fi

# Run the application
echo "Starting CUDA image processor..."
./build/image_processor_cuda

echo "Job completed!"

