# AAST Cluster Build and Run Guide

This guide is specific to the AAST cluster setup with `submit.nvcc` and `submit.gpu`.

## Quick Start

### 1. Pull Latest Code
```bash
cd ~/HighPerformanceImageProcessing
git checkout cuda
git pull origin cuda
```

### 2. Build
```bash
chmod +x build_cluster_aast.sh
./build_cluster_aast.sh
```

### 3. Submit Job

**Basic submission:**
```bash
sbatch submit.gpu "./build/image_processor_cuda"
```

**With timing:**
```bash
sbatch submit.gpu.timed "./build/image_processor_cuda"
```

**With profiling:**
```bash
sbatch submit.gpu.profile "./build/image_processor_cuda"
```

## Cluster-Specific Commands

### Compile CUDA Code
```bash
# Single file
submit.nvcc src/cuda_kernels.cu -o cuda_kernels.o -Iinclude

# With math library
submit.nvcc src/cuda_kernels.cu -o cuda_kernels.o -Iinclude -lm
```

### Submit GPU Job
```bash
# Basic
sbatch submit.gpu "./build/image_processor_cuda"

# With timing
sbatch submit.gpu.timed "./build/image_processor_cuda"

# With profiling
sbatch submit.gpu.profile "./build/image_processor_cuda"
```

### Monitor Jobs
```bash
# Watch job queue
watch -n 1 "squeue"

# Check your jobs
squeue -u $USER
```

## Manual Build (Alternative)

If the automated script doesn't work, build manually:

```bash
# Compile CUDA kernels
submit.nvcc -c src/cuda_kernels.cu -o build/cuda_kernels.o \
    -Iinclude -Ilib -arch=sm_70 -O3 -std=c++17

# Compile C++ files
g++ -c src/main_cuda.cpp -o build/main_cuda.o \
    -Iinclude -Ilib -O3 -std=c++17

g++ -c src/cuda_utils.cpp -o build/cuda_utils.o \
    -Iinclude -Ilib -O3 -std=c++17

g++ -c src/filters_cuda.cpp -o build/filters_cuda.o \
    -Iinclude -Ilib -O3 -std=c++17

g++ -c src/edge_detection_cuda.cpp -o build/edge_detection_cuda.o \
    -Iinclude -Ilib -O3 -std=c++17

# Compile other C++ files (image.cpp, point_operations.cpp, etc.)
# ... (similar pattern)

# Link everything
g++ build/*.o -o build/image_processor_cuda -lcudart -lm
```

## Using CMake (If Available)

If CMake works on the cluster:

```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON \
         -DCMAKE_CUDA_COMPILER=submit.nvcc \
         -DCMAKE_CXX_COMPILER=g++
make -j4
cd ..
```

## Job Submission Examples

### Example 1: Process an Image
```bash
# The application is interactive, so you might need to create an input file
# Or modify the code to accept command-line arguments

sbatch submit.gpu "./build/image_processor_cuda"
```

### Example 2: With Timing
```bash
sbatch submit.gpu.timed "./build/image_processor_cuda"
```

### Example 3: With Performance Profiling
```bash
sbatch submit.gpu.profile "./build/image_processor_cuda"
```

### Example 4: With Input File (if modified)
```bash
sbatch submit.gpu "./build/image_processor_cuda" "input/set1.png"
```

## Check Job Status

```bash
# Monitor all jobs
watch -n 1 "squeue"

# Check your jobs only
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Cancel job
scancel <job_id>
```

## View Output

After job completes:
```bash
# View output
cat output_<job_id>.out

# View errors
cat error_<job_id>.err
```

## Troubleshooting

### submit.nvcc not found
```bash
# Check if it's in PATH
which submit.nvcc

# Check if it's a script
ls -la $(which submit.nvcc)
```

### Build fails
- Check CUDA architecture: `submit.nvcc --version`
- Adjust `-arch=sm_XX` flag in build script to match your GPU
- Common architectures: sm_60, sm_70, sm_75, sm_80

### Job fails
- Check error file: `cat error_<job_id>.err`
- Verify GPU is available: Check if `--gres=gpu:1` is correct
- Check time limit: Increase `--time` if needed

## Quick Reference

```bash
# Build
./build_cluster_aast.sh

# Submit
sbatch submit.gpu "./build/image_processor_cuda"

# Monitor
watch -n 1 "squeue"

# Check output
cat output_*.out
```

