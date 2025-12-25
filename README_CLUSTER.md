# Quick Start: Building and Running on HPC Cluster

## Quick Build Steps

### 1. Load Modules
```bash
module load cuda/11.8 gcc/9.4.0 cmake/3.20
```

### 2. Build
```bash
./build_cluster.sh
```

Or manually:
```bash
mkdir build && cd build
cmake .. -DENABLE_CUDA=ON
make -j4
cd ..
```

### 3. Test Interactively (if you have GPU access)
```bash
# Request interactive GPU node (SLURM)
srun --gres=gpu:1 --time=01:00:00 --pty bash

# Once on GPU node
./build/image_processor_cuda
```

### 4. Submit as Job
```bash
# SLURM
sbatch submit_cuda.sh

# PBS/Torque
qsub submit_cuda.pbs
```

## Check GPU Availability

```bash
# Check if GPU is available
nvidia-smi

# Check CUDA installation
nvcc --version
```

## Common Commands

```bash
# Find available CUDA modules
module avail cuda

# Check job status (SLURM)
squeue -u $USER

# Check job status (PBS)
qstat -u $USER

# Cancel job
scancel <job_id>    # SLURM
qdel <job_id>       # PBS
```

## Troubleshooting

1. **Can't find nvcc**: Load CUDA module
   ```bash
   module load cuda/11.8
   ```

2. **CMake can't find CUDA**: Set CUDA_HOME
   ```bash
   export CUDA_HOME=/usr/local/cuda-11.8
   ```

3. **Wrong GPU architecture**: Check your GPU compute capability
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

For detailed information, see `docs/CLUSTER_BUILD_GUIDE.md`

