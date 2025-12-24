# OpenMP Implementation Guide

## Overview

This document describes the OpenMP parallelization implemented in the image processing project. The OpenMP version leverages shared-memory parallelism to accelerate computationally intensive image processing operations across multiple CPU cores.

## Branch Information

- **Branch Name**: `openmp`
- **Base Branch**: `main`
- **Parallelization Model**: Shared-memory parallelism using OpenMP

## Compilation Requirements

### Compiler Support
- C++17 compatible compiler with OpenMP support
- GCC 4.9+ or Clang 3.7+ or MSVC 2019+ recommended

### Build Flags
```bash
# Using g++
g++ -std=c++17 -fopenmp -O2 -I./include -I./lib src/*.cpp -o image_processor

# Using CMake (if available)
cmake .. -DCMAKE_CXX_FLAGS="-fopenmp"
make

# Using Makefile
make  # OpenMP flags already included in Makefile
```

## Parallelization Strategy

### 1. Loop-Level Parallelization

Most image processing operations involve iterating over all pixels in an image. These loops are ideal candidates for parallelization since each pixel can be processed independently.

**Pattern Used:**
```cpp
#pragma omp parallel for collapse(2)
for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
        // Process pixel at (x, y)
    }
}
```

The `collapse(2)` clause combines the nested loops into a single iteration space, allowing better load balancing across threads.

### 2. Data-Level Parallelization

For operations on flat arrays (like brightness adjustment), simple parallel for loops are used:

```cpp
#pragma omp parallel for
for (size_t i = 0; i < size; ++i) {
    // Process data[i]
}
```

### 3. Reduction Operations

For histogram computation (used in Otsu thresholding), thread-local storage is used to avoid race conditions:

```cpp
#pragma omp parallel
{
    int local_hist[256] = {0};
    #pragma omp for nowait
    for (size_t i = 0; i < size; ++i) {
        local_hist[data[i]]++;
    }
    #pragma omp critical
    {
        for (int i = 0; i < 256; ++i) {
            histogram[i] += local_hist[i];
        }
    }
}
```

### 4. Thread-Safe Random Number Generation

For noise generation functions, each thread gets its own random number generator:

```cpp
#pragma omp parallel
{
    std::mt19937 rng(seed + omp_get_thread_num());
    std::normal_distribution<float> dist(mean, stddev);
    
    #pragma omp for
    for (size_t i = 0; i < size; ++i) {
        // Generate noise using thread-local rng
    }
}
```

## Parallelized Operations

### Point Operations (`src/point_operations.cpp`)
- ✅ `grayscale()` - Parallel pixel-wise conversion
- ✅ `adjustBrightness()` - Parallel array processing
- ✅ `adjustContrast()` - Parallel array processing
- ✅ `threshold()` - Parallel thresholding
- ✅ `thresholdOtsu()` - Parallel histogram computation
- ✅ `adaptiveThreshold()` - Parallel with local window computation
- ✅ `adaptiveThresholdNiblack()` - Parallel adaptive thresholding
- ✅ `adaptiveThresholdSauvola()` - Parallel adaptive thresholding
- ✅ `invert()` - Parallel array processing
- ✅ `gammaCorrection()` - Parallel LUT application

### Filters (`src/filters.cpp`)
- ✅ `boxBlur()` - Parallel convolution
- ✅ `gaussianBlur()` - Parallel convolution
- ✅ `medianFilter()` - Parallel median computation
- ✅ `bilateralFilter()` - Parallel edge-preserving filter

### Edge Detection (`src/edge_detection.cpp`)
- ✅ `sobelX()` - Parallel gradient computation
- ✅ `sobelY()` - Parallel gradient computation
- ✅ `sobel()` - Parallel edge detection
- ✅ `canny()` - Multi-stage parallel edge detection
- ✅ `sharpen()` - Parallel sharpening filter
- ✅ `prewitt()` - Parallel edge detection
- ✅ `laplacian()` - Parallel edge detection

### Geometric Transformations (`src/geometric.cpp`)
- ✅ `rotate()` - Parallel rotation with interpolation
- ✅ `resize()` - Parallel scaling with interpolation
- ✅ `translate()` - Parallel translation
- ✅ `flipHorizontal()` - Parallel horizontal flip
- ✅ `flipVertical()` - Parallel vertical flip

### Morphological Operations (`src/morphological.cpp`)
- ✅ `erode()` - Parallel erosion
- ✅ `dilate()` - Parallel dilation
- ✅ `morphologicalGradient()` - Parallel gradient computation
- ⚠️ `opening()` - Sequential (calls parallelized functions)
- ⚠️ `closing()` - Sequential (calls parallelized functions)

### Noise Operations (`src/noise.cpp`)
- ⚠️ `saltAndPepper()` - Not parallelized (random access pattern)
- ✅ `gaussian()` - Parallel with thread-local RNG
- ✅ `speckle()` - Parallel with thread-local RNG

### Color Operations (`src/color_operations.cpp`)
- ✅ `splitChannels()` - Parallel channel extraction
- ✅ `mergeChannels()` - Parallel channel combination
- ✅ `rgbToHsv()` - Parallel color space conversion
- ✅ `hsvToRgb()` - Parallel color space conversion
- ⚠️ `adjustHue()` - Sequential (calls parallelized functions)
- ⚠️ `adjustSaturation()` - Sequential (calls parallelized functions)
- ⚠️ `adjustValue()` - Sequential (calls parallelized functions)
- ✅ `colorBalance()` - Parallel color adjustment
- ✅ `toneMapping()` - Parallel tone mapping

## Performance Considerations

### Expected Speedup

The theoretical speedup depends on:
1. **Number of CPU cores**: More cores = better parallelization
2. **Image size**: Larger images benefit more from parallelization
3. **Operation complexity**: Complex operations (like bilateral filter) see better speedup
4. **Memory bandwidth**: Can become a bottleneck for simple operations

**Typical speedup ranges:**
- Simple operations (brightness, contrast): 2-4x on 4 cores
- Medium operations (Gaussian blur, Sobel): 3-6x on 4 cores
- Complex operations (bilateral filter, Canny): 4-8x on 4 cores

### Overhead Considerations

OpenMP introduces some overhead:
- Thread creation and synchronization
- Work distribution
- Memory access patterns

For very small images (< 100x100 pixels), the overhead may outweigh the benefits.

### Thread Count Control

You can control the number of threads using:

```bash
# Set number of threads via environment variable
export OMP_NUM_THREADS=4

# Or in code (before parallel regions)
omp_set_num_threads(4);
```

## Testing and Validation

### Correctness Testing

To verify the OpenMP version produces identical results to the serial version:

1. Process the same image with both versions
2. Compare output pixel-by-pixel
3. Ensure numerical differences are within acceptable tolerance (due to floating-point rounding)

### Performance Testing

To measure speedup:

```bash
# Time a specific operation
time ./image_processor < test_commands.txt

# Compare with serial version
git checkout main
make clean && make
time ./image_processor < test_commands.txt
```

## Common Issues and Solutions

### Issue 1: Race Conditions
**Symptom**: Inconsistent or incorrect results
**Solution**: Ensure no shared data is written by multiple threads simultaneously. Use `#pragma omp critical` or thread-local storage.

### Issue 2: False Sharing
**Symptom**: Poor performance despite parallelization
**Solution**: Ensure threads work on different cache lines. Use padding or restructure data access patterns.

### Issue 3: Load Imbalance
**Symptom**: Some threads finish much earlier than others
**Solution**: Use `schedule(dynamic)` or `schedule(guided)` clauses for better load balancing.

### Issue 4: Nested Parallelism
**Symptom**: Performance degradation when parallel functions call other parallel functions
**Solution**: Disable nested parallelism or carefully control thread counts at each level.

## Future Optimizations

Potential improvements for the OpenMP version:

1. **SIMD Vectorization**: Add `#pragma omp simd` for inner loops
2. **Task-Based Parallelism**: Use OpenMP tasks for irregular workloads
3. **Memory Optimization**: Improve cache locality and reduce false sharing
4. **Hybrid Parallelism**: Combine with SIMD intrinsics for maximum performance
5. **Adaptive Threading**: Dynamically adjust thread count based on image size

## Comparison with Other Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **OpenMP** | Easy to implement, portable, good for shared memory | Limited to single node, memory bandwidth bound |
| **MPI** | Scales to multiple nodes, good for large images | More complex, communication overhead |
| **CUDA** | Massive parallelism, very fast for suitable operations | Requires NVIDIA GPU, more complex code |
| **OpenCL** | Portable across GPUs, similar to CUDA | Complex API, device-specific tuning needed |

## References

- [OpenMP Specification](https://www.openmp.org/specifications/)
- [OpenMP Best Practices](https://www.openmp.org/resources/tutorials-articles/)
- [Image Processing Parallelization Patterns](https://www.intel.com/content/www/us/en/developer/articles/technical/image-processing-using-openmp.html)

## License

Same as the main project (MIT License)

