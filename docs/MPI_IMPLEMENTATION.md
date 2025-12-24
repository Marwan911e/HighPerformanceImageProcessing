# MPI Implementation Guide

## Overview

The MPI branch implements distributed-memory parallelization using Message Passing Interface (MPI). This allows the image processing application to run across multiple nodes/machines in a cluster environment.

## Architecture

### Image Distribution Strategy

- **Horizontal Strip Decomposition**: The image is divided into horizontal strips
- Each MPI process receives a portion of rows
- Rank 0 handles I/O (loading/saving) and user interaction
- All processes work on their local chunk in parallel

### Communication Patterns

1. **Image Distribution**: Rank 0 loads image, then distributes chunks to all processes
2. **Operation Processing**: Each process works on its local chunk (embarrassingly parallel for point operations)
3. **Result Gathering**: All processes send their results back to rank 0 for saving

## Supported Operations

### Currently Implemented (Embarrassingly Parallel)

These operations work independently on each chunk - no communication needed:

- ✅ **Grayscale** (option 10)
- ✅ **Brightness Adjustment** (option 11)
- ✅ **Contrast Adjustment** (option 12)
- ✅ **Threshold** (option 13)
- ✅ **Invert** (option 16)

### Future Implementations

Operations requiring boundary exchange (for filters):

- ⏳ **Gaussian Blur** - Needs neighbor pixels
- ⏳ **Box Blur** - Needs neighbor pixels
- ⏳ **Sobel Edge Detection** - Needs neighbor pixels
- ⏳ **Median Filter** - Needs neighbor pixels

## Building

### Prerequisites

- MPI implementation (OpenMPI, MPICH, or Intel MPI)
- C++17 compatible compiler
- STB Image library (included)

### Compilation

```bash
# Make sure you're on MPI branch
git checkout mpi

# Build
make clean
make
```

The Makefile uses `mpic++` compiler wrapper which automatically includes MPI headers and libraries.

### Manual Compilation

```bash
mpic++ -std=c++17 -O2 -I./include -I./lib \
    src/*.cpp -o build/image_processor
```

## Running

### Basic Usage

```bash
# Run with 4 processes
mpirun -np 4 ./build/image_processor

# Or using srun (SLURM)
srun -n 4 ./build/image_processor
```

### On Cluster (SLURM)

Create a job script `run_mpi.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=image_mpi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=mpi_output_%j.log

cd ~/hpc2final
mpirun -np 8 ./build/image_processor
```

Submit job:
```bash
sbatch run_mpi.sh
```

### Interactive Run

```bash
# SSH into cluster
ssh m.ahmed03166@student.aast.edu

# Navigate to project
cd ~/hpc2final

# Run with 8 processes
mpirun -np 8 ./build/image_processor
```

## Performance Considerations

### When to Use MPI

- ✅ **Large images** (>10K x 10K pixels)
- ✅ **Multiple nodes available**
- ✅ **Memory constraints** (distribute across nodes)
- ✅ **Cluster environment**

### When NOT to Use MPI

- ❌ Single machine (use OpenMP instead)
- ❌ Small images (<2K x 2K)
- ❌ Network latency overhead too high

### Expected Speedup

- **Ideal**: Linear speedup with number of processes (for embarrassingly parallel ops)
- **Realistic**: 0.7-0.9x efficiency due to communication overhead
- **Example**: 8 processes on 8-core image → ~6-7x speedup

## Code Structure

### Key Files

- `include/mpi_utils.h` - MPI helper functions
- `src/mpi_utils.cpp` - Image distribution/gathering
- `include/point_operations_mpi.h` - MPI operation wrappers
- `src/point_operations_mpi.cpp` - MPI operation implementations
- `src/main.cpp` - Modified for MPI execution model

### MPI Functions

```cpp
// Distribute full image to all processes
MPIUtils::distributeImage(fullImage, localChunk, rank, size);

// Gather chunks back to rank 0
MPIUtils::gatherImage(localChunk, fullImage, rank, size);

// Exchange boundaries (for filters - future)
MPIUtils::exchangeBoundaries(localChunk, rank, size, boundarySize);
```

## Limitations

1. **I/O Only on Rank 0**: All file operations must happen on rank 0
2. **Menu Interaction**: Only rank 0 displays menu and gets user input
3. **Limited Operations**: Currently only point operations implemented
4. **No Boundary Exchange**: Filter operations not yet implemented (need ghost cells)

## Future Enhancements

1. **Boundary Exchange**: Implement ghost cells for filter operations
2. **More Operations**: Add MPI versions of filters and edge detection
3. **Load Balancing**: Better distribution for non-divisible image sizes
4. **Hybrid MPI+OpenMP**: Use MPI across nodes, OpenMP within nodes
5. **Parallel I/O**: Distribute file reading/writing across processes

## Troubleshooting

### "mpic++ not found"

```bash
# Load MPI module (cluster-specific)
module load openmpi
# or
module load mpich
```

### "Cannot open output file"

- Make sure only rank 0 saves files
- Check file permissions
- Verify output directory exists

### "Hanging/Deadlock"

- Check that all processes participate in MPI operations
- Verify MPI_Bcast calls match on all ranks
- Use debugger: `mpirun -np 4 gdb ./build/image_processor`

### Performance Issues

- Profile with: `mpirun -np 4 perf ./build/image_processor`
- Check network bandwidth between nodes
- Verify load balancing (image size divisible by process count)

## Comparison with Other Branches

| Feature | Serial | OpenMP | MPI |
|---------|--------|--------|-----|
| **Memory** | Shared | Shared | Distributed |
| **Best For** | Single core | Multi-core | Multi-node |
| **Complexity** | Low | Medium | High |
| **Operations** | All | All | Point ops only |
| **Setup** | Easy | Easy | Medium |

## Example Workflow

```bash
# 1. Build
make

# 2. Transfer image to cluster
scp image.jpg m.ahmed03166@student.aast.edu:~/hpc2final/input/

# 3. Run on cluster
ssh m.ahmed03166@student.aast.edu
cd ~/hpc2final
mpirun -np 8 ./build/image_processor

# 4. In application:
#    - Select image (option 1)
#    - Apply grayscale (option 10)
#    - Apply brightness (option 11)
#    - Save (option 2)

# 5. Transfer results back
scp m.ahmed03166@student.aast.edu:~/hpc2final/output/*.jpg ./
```

## References

- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
- [MPICH Documentation](https://www.mpich.org/documentation/)
- [MPI Standard](https://www.mpi-forum.org/docs/)

---

**Note**: This is a work-in-progress implementation. More operations will be added over time.

