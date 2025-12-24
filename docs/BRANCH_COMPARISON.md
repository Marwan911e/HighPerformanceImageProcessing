# Branch Comparison: Main vs OpenMP

## Overview

This document provides a quick comparison between the `main` (serial) branch and the `openmp` (parallel) branch.

## Branch Summary

### Main Branch (Serial Version)
- **Purpose**: Baseline serial implementation
- **Performance**: Single-threaded execution
- **Use Case**: Reference implementation, small images, debugging
- **Compilation**: Standard C++17 compiler

### OpenMP Branch (Parallel Version)
- **Purpose**: Shared-memory parallel implementation
- **Performance**: Multi-threaded execution (scales with CPU cores)
- **Use Case**: Large images, production workloads, performance-critical applications
- **Compilation**: C++17 compiler with OpenMP support

## Key Differences

### 1. Code Changes

| File | Changes in OpenMP Branch |
|------|-------------------------|
| `src/point_operations.cpp` | Added `#include <omp.h>` and `#pragma omp` directives to 9 functions |
| `src/filters.cpp` | Added `#include <omp.h>` and `#pragma omp` directives to 4 functions |
| `src/edge_detection.cpp` | Added `#include <omp.h>` and `#pragma omp` directives to 9 functions |
| `src/geometric.cpp` | Added `#include <omp.h>` and `#pragma omp` directives to 5 functions |
| `src/morphological.cpp` | Added `#include <omp.h>` and `#pragma omp` directives to 3 functions |
| `src/noise.cpp` | Added `#include <omp.h>` and thread-safe RNG to 2 functions |
| `src/color_operations.cpp` | Added `#include <omp.h>` and `#pragma omp` directives to 6 functions |
| `src/main.cpp` | Changed banner from "SERIAL" to "OpenMP" |
| `Makefile` | Added `-fopenmp` flags to CXXFLAGS and LDFLAGS |
| `CMakeLists.txt` | No changes (already had OpenMP support) |
| `README.md` | Updated to reflect OpenMP implementation |

### 2. Performance Characteristics

#### Serial Version (Main Branch)
```
Single-threaded execution
Predictable performance
Lower memory usage
Easier to debug
```

#### OpenMP Version (OpenMP Branch)
```
Multi-threaded execution
Performance scales with cores
Higher memory usage (thread stacks)
Potential for race conditions (if bugs exist)
```

### 3. Typical Performance Comparison

Assuming a 4-core CPU and a 2048x2048 pixel image:

| Operation | Serial Time | OpenMP Time | Speedup |
|-----------|-------------|-------------|---------|
| Grayscale | 10 ms | 3 ms | 3.3x |
| Gaussian Blur (5x5) | 150 ms | 40 ms | 3.8x |
| Sobel Edge Detection | 80 ms | 22 ms | 3.6x |
| Bilateral Filter | 500 ms | 140 ms | 3.6x |
| Image Rotation | 120 ms | 35 ms | 3.4x |
| Median Filter (5x5) | 300 ms | 85 ms | 3.5x |

*Note: Actual performance depends on CPU, memory, and image characteristics*

### 4. Memory Usage

| Aspect | Serial | OpenMP |
|--------|--------|--------|
| Base memory | X | X |
| Thread stacks | 0 | ~1-8 MB per thread |
| Thread-local data | 0 | Varies by operation |
| Total overhead | Minimal | ~4-32 MB for 4-8 threads |

### 5. Compilation and Execution

#### Serial Version
```bash
# Compile
g++ -std=c++17 -O2 -I./include -I./lib src/*.cpp -o image_processor

# Run
./image_processor
```

#### OpenMP Version
```bash
# Compile
g++ -std=c++17 -O2 -fopenmp -I./include -I./lib src/*.cpp -o image_processor

# Run with 4 threads
export OMP_NUM_THREADS=4
./image_processor

# Or
OMP_NUM_THREADS=4 ./image_processor
```

## When to Use Each Version

### Use Serial Version (Main Branch) When:
- ✅ Debugging code
- ✅ Processing small images (< 512x512)
- ✅ Running on single-core systems
- ✅ Memory is extremely limited
- ✅ Need deterministic execution order
- ✅ Profiling individual operations

### Use OpenMP Version (OpenMP Branch) When:
- ✅ Processing large images (> 1024x1024)
- ✅ Running on multi-core systems (2+ cores)
- ✅ Performance is critical
- ✅ Batch processing multiple images
- ✅ Real-time or near-real-time processing needed
- ✅ Have sufficient memory (RAM)

## Switching Between Branches

```bash
# Switch to serial version
git checkout main
make clean && make

# Switch to OpenMP version
git checkout openmp
make clean && make

# Compare branches
git diff main openmp

# View branch-specific files
git show openmp:OPENMP_IMPLEMENTATION.md
```

## Testing Both Versions

To ensure both versions produce identical results:

```bash
# Process image with serial version
git checkout main
make
./image_processor
# (apply operations and save as output_serial.jpg)

# Process same image with OpenMP version
git checkout openmp
make
./image_processor
# (apply same operations and save as output_openmp.jpg)

# Compare outputs (should be identical or nearly identical)
# Use image comparison tools or compute PSNR/SSIM
```

## Merging Strategy

If you need to merge changes from main to openmp:

```bash
# Update openmp branch with changes from main
git checkout openmp
git merge main

# Resolve any conflicts (usually in non-parallelized code)
# Test thoroughly after merge
```

## Future Branches

Potential additional branches:
- `mpi` - Distributed-memory parallelization for clusters
- `cuda` - GPU acceleration using NVIDIA CUDA
- `opencl` - GPU acceleration using OpenCL (portable)
- `hybrid` - Combination of OpenMP + MPI or OpenMP + CUDA

## Questions?

See [OPENMP_IMPLEMENTATION.md](OPENMP_IMPLEMENTATION.md) for detailed OpenMP documentation.

## Summary

| Aspect | Main (Serial) | OpenMP (Parallel) |
|--------|--------------|-------------------|
| **Speed** | Baseline | 2-8x faster |
| **Complexity** | Simple | Moderate |
| **Portability** | High | High (with OpenMP support) |
| **Memory** | Low | Medium |
| **Best For** | Small images, debugging | Large images, production |

