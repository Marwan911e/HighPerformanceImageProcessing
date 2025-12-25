# Building and Running CUDA Implementation on HPC Cluster

This guide explains how to build and run the CUDA image processing application on an HPC cluster.

## Prerequisites

1. Access to a cluster node with CUDA-capable GPU
2. CUDA Toolkit installed on the cluster
3. CMake 3.10 or later
4. C++17 compatible compiler (gcc/clang)

## Step 1: Load Required Modules

Most clusters use environment modules. Load CUDA and compiler modules:

```bash
# Example for most clusters (adjust module names as needed)
module load cuda/11.8          # or cuda/12.0, check available versions
module load gcc/9.4.0           # or your cluster's default compiler
module load cmake/3.20          # if CMake is not in default path
```

**To find available modules:**
```bash
module avail cuda
module avail gcc
module avail cmake
```

## Step 2: Verify CUDA Installation

```bash
# Check CUDA compiler
nvcc --version

# Check GPU availability (on compute nodes)
nvidia-smi

# Check CUDA environment variables
echo $CUDA_HOME
echo $CUDA_PATH
```

## Step 3: Build the Application

### Option A: Using CMake (Recommended)

```bash
# Navigate to project directory
cd /path/to/hpc2final/hpc2final

# Create build directory
mkdir build && cd build

# Configure with CUDA enabled
cmake .. -DENABLE_CUDA=ON \
         -DCMAKE_CUDA_COMPILER=nvcc \
         -DCMAKE_CXX_COMPILER=g++ \
         -DCMAKE_BUILD_TYPE=Release

# Build
make -j4  # Adjust -j based on available cores

# Or use cmake build
cmake --build . -j4
```

### Option B: Manual Build (If CMake Issues)

Create a `Makefile.cluster`:

```makefile
NVCC = nvcc
CXX = g++
CUDA_PATH = $(CUDA_HOME)
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
CUDA_INC_PATH = $(CUDA_PATH)/include

NVCC_FLAGS = -arch=sm_70 -O3 -std=c++17 --compiler-options -fPIC
CXX_FLAGS = -O3 -std=c++17 -fPIC

# Source files
CUDA_SOURCES = src/cuda_kernels.cu
CPP_SOURCES = src/main_cuda.cpp src/image.cpp src/cuda_utils.cpp \
              src/filters_cuda.cpp src/edge_detection_cuda.cpp \
              src/point_operations.cpp src/noise.cpp \
              src/morphological.cpp src/geometric.cpp src/color_operations.cpp

# Object files
CUDA_OBJS = $(CUDA_SOURCES:.cu=.o)
CPP_OBJS = $(CPP_SOURCES:.cpp=.o)

# Include directories
INCLUDES = -Iinclude -Ilib -I$(CUDA_INC_PATH)

# Libraries
LIBS = -L$(CUDA_LIB_PATH) -lcudart -lcuda

# Target
TARGET = image_processor_cuda

all: $(TARGET)

$(TARGET): $(CUDA_OBJS) $(CPP_OBJS)
	$(CXX) $(CPP_OBJS) $(CUDA_OBJS) -o $(TARGET) $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(CUDA_OBJS) $(CPP_OBJS)
```

Build with:
```bash
make -f Makefile.cluster
```

## Step 4: Submit Job to Cluster

### SLURM Job Script Example

Create `submit_cuda.sh`:

```bash
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

# Load modules
module load cuda/11.8
module load gcc/9.4.0

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run the application
./build/image_processor_cuda
```

**Submit job:**
```bash
sbatch submit_cuda.sh
```

### PBS/Torque Job Script Example

Create `submit_cuda.pbs`:

```bash
#!/bin/bash
#PBS -N image_process_cuda
#PBS -o output_%j.out
#PBS -e error_%j.err
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l mem=8gb
#PBS -q gpu

# Load modules
module load cuda/11.8
module load gcc/9.4.0

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd $PBS_O_WORKDIR

# Run the application
./build/image_processor_cuda
```

**Submit job:**
```bash
qsub submit_cuda.pbs
```

## Step 5: Interactive GPU Session (For Testing)

### SLURM Interactive Session

```bash
# Request interactive GPU node
srun --job-name=test_cuda \
     --time=01:00:00 \
     --gres=gpu:1 \
     --partition=gpu \
     --mem=8GB \
     --pty bash

# Once on the node, load modules and run
module load cuda/11.8 gcc/9.4.0
cd /path/to/project
./build/image_processor_cuda
```

### PBS Interactive Session

```bash
# Request interactive GPU node
qsub -I -l nodes=1:ppn=4:gpus=1 -l walltime=01:00:00 -q gpu

# Once on the node, load modules and run
module load cuda/11.8 gcc/9.4.0
cd /path/to/project
./build/image_processor_cuda
```

## Common Issues and Solutions

### Issue 1: CUDA Not Found

```bash
# Check if CUDA_HOME is set
echo $CUDA_HOME

# If not set, find CUDA installation
find /usr/local -name "nvcc" 2>/dev/null
find /opt -name "nvcc" 2>/dev/null

# Set manually if needed
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Issue 2: Wrong GPU Architecture

Check your GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Update CMakeLists.txt or Makefile with correct architecture:
```cmake
# In CMakeLists.txt, add:
set(CMAKE_CUDA_ARCHITECTURES "70;75;80")  # Adjust for your GPU
```

Or in Makefile:
```makefile
NVCC_FLAGS = -arch=sm_75 -O3 ...  # Change sm_75 to your GPU's compute capability
```

Common compute capabilities:
- sm_60: Pascal (GTX 1080, P100)
- sm_70: Volta (V100)
- sm_75: Turing (RTX 2080, T4)
- sm_80: Ampere (A100, RTX 3090)
- sm_86: Ampere (RTX 3080, 3090)
- sm_89: Ada Lovelace (RTX 4090)

### Issue 3: CMake Can't Find CUDA

```bash
# Specify CUDA path explicitly
cmake .. -DENABLE_CUDA=ON \
         -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
         -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc
```

### Issue 4: Permission Denied

```bash
# Make sure executable has execute permission
chmod +x build/image_processor_cuda

# Check if you're on a GPU node
nvidia-smi
```

### Issue 5: Out of Memory

Reduce image size or use tiled processing. Check GPU memory:
```bash
nvidia-smi
```

## Testing on Cluster

### Quick Test Script

Create `test_cuda.sh`:

```bash
#!/bin/bash
module load cuda/11.8 gcc/9.4.0

cd /path/to/project/build

# Test with a small image
echo "1" | ./image_processor_cuda  # Select image
echo "30" | ./image_processor_cuda  # Box blur
echo "5" | ./image_processor_cuda   # Kernel size 5
echo "2" | ./image_processor_cuda   # Save
```

### Batch Processing Script

Create `batch_process.sh`:

```bash
#!/bin/bash
# Process multiple images

module load cuda/11.8 gcc/9.4.0

cd /path/to/project/build

for img in ../input/*.png; do
    echo "Processing $img"
    # Your processing commands here
done
```

## Performance Monitoring

### Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Or in a separate terminal
nvidia-smi -l 1
```

### Profile with nvprof

```bash
nvprof ./build/image_processor_cuda
```

## Cluster-Specific Notes

### For Specific Clusters:

**NERSC (Perlmutter, Cori):**
```bash
module load cudatoolkit
module load gcc
```

**Purdue (Brown):**
```bash
module load cuda/11.8.0
module load gcc/9.4.0
```

**TACC (Stampede2, Frontera):**
```bash
module load cuda/11.0
module load gcc
```

**Check your cluster's documentation for:**
- Available GPU partitions
- Module names
- Job scheduler commands
- GPU node specifications

## Environment Variables

Add to your `~/.bashrc` or job script:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

## Quick Reference

```bash
# Build
cd build && cmake .. -DENABLE_CUDA=ON && make -j4

# Test interactively
srun --gres=gpu:1 --pty bash
./build/image_processor_cuda

# Submit job
sbatch submit_cuda.sh

# Check job status
squeue -u $USER

# Monitor GPU
nvidia-smi
```

