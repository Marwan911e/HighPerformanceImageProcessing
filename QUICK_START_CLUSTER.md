# Quick Start Guide for Cluster (No Module System)

If your cluster doesn't use environment modules, follow these steps:

## Step 1: Check CUDA Availability

```bash
./check_cuda.sh
```

Or manually:
```bash
# Check if nvcc is available
which nvcc
nvcc --version

# If not found, search for CUDA
find /usr/local /opt -name nvcc 2>/dev/null
```

## Step 2: Set CUDA_HOME (if needed)

If CUDA is found but not in PATH:

```bash
# Example - adjust path to your CUDA installation
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Step 3: Build

### Option A: Using the Simple Build Script (Recommended)

```bash
chmod +x build_cluster_simple.sh
./build_cluster_simple.sh
```

### Option B: Using CMake Manually

```bash
mkdir build && cd build

# If CUDA_HOME is set
cmake .. -DENABLE_CUDA=ON \
         -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
         -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc

# Or if nvcc is in PATH
cmake .. -DENABLE_CUDA=ON

make -j4
cd ..
```

### Option C: Using Makefile (No CMake)

```bash
# Set CUDA_HOME if needed
export CUDA_HOME=/usr/local/cuda-11.8

# Build
make -f Makefile.cluster
```

## Step 4: Run

### Interactive (if on GPU node)

```bash
# Check GPU
nvidia-smi

# Run
./build/image_processor_cuda
# or if using Makefile
./image_processor_cuda
```

### Submit as Job

Create a job script (adjust for your cluster's scheduler):

**For SLURM:**
```bash
#!/bin/bash
#SBATCH --job-name=cuda_image
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Set CUDA if needed
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd $SLURM_SUBMIT_DIR
./build/image_processor_cuda
```

**For PBS:**
```bash
#!/bin/bash
#PBS -N cuda_image
#PBS -l nodes=1:gpus=1
#PBS -l walltime=01:00:00

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd $PBS_O_WORKDIR
./build/image_processor_cuda
```

## Troubleshooting

### CUDA Not Found

```bash
# Find CUDA installation
find /usr/local -name nvcc 2>/dev/null
find /opt -name nvcc 2>/dev/null

# Set CUDA_HOME to the directory containing 'bin/nvcc'
export CUDA_HOME=/path/to/cuda
```

### CMake Can't Find CUDA

```bash
# Specify CUDA path explicitly
cmake .. -DENABLE_CUDA=ON \
         -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc
```

### Wrong GPU Architecture

Check your GPU:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Update architecture in CMakeLists.txt or Makefile:
- sm_60: Pascal
- sm_70: Volta  
- sm_75: Turing
- sm_80: Ampere

### Permission Denied

```bash
chmod +x build_cluster_simple.sh check_cuda.sh
chmod +x build/image_processor_cuda
```

## Quick Commands

```bash
# Check setup
./check_cuda.sh

# Build
./build_cluster_simple.sh

# Or with Makefile
make -f Makefile.cluster

# Run
./build/image_processor_cuda
```

