# CUDA Implementation

This branch implements image processing operations using CUDA with advanced optimizations.

## Features

### 1. Pinned (Page-Locked) Host Memory
- Uses `cudaHostAlloc()` to allocate pinned host buffers
- Enables faster host-to-device and device-to-host transfers
- Implemented in `CudaUtils::PinnedBuffer` class

### 2. Asynchronous Transfers & Streams
- Uses `cudaMemcpyAsync()` for non-blocking memory transfers
- Multiple CUDA streams for pipelining operations
- Implemented via `CudaUtils::StreamManager` class
- Overlaps H2D/D2H copies with kernel execution

### 3. Overlap Strategy with Tiling
- Splits images into tiles for processing
- While GPU works on tile N:
  - Transfer tile N+1 to device (async)
  - Transfer tile N-1 back to host (async)
- Implemented in `CudaUtils::generateTiles()` and tiled filter functions

### 4. Shared Memory for Convolution Tiles
- Each block loads its tile + border pixels into shared memory
- Reduces global memory reads significantly
- Implemented in CUDA kernels:
  - `boxBlurKernel` - uses shared memory with halo regions
  - `gaussianBlurKernel` - uses shared memory + constant memory for kernel
  - `sobelKernel` - uses shared memory for edge detection

### 5. Filter Coefficients in Constant Memory
- Small read-only filter kernels stored in `__constant__` memory
- Fast, cached access for filter coefficients
- Example: Gaussian kernel stored in `c_gaussianKernel` constant memory

### 6. Memory Transfer Minimization
- Keeps intermediate buffers on GPU when chaining filters
- Only copies back to host when final image is ready
- Reduces unnecessary round trips between host and device

## Building

### Prerequisites
- CUDA Toolkit (version 11.0 or later recommended)
- CMake 3.10 or later
- C++17 compatible compiler

### Build Instructions

```bash
# Create build directory
mkdir build && cd build

# Configure with CUDA enabled
cmake .. -DENABLE_CUDA=ON

# Build
cmake --build .

# Run
./image_processor_cuda
```

## Implementation Details

### Kernel Architecture

#### Box Blur Kernel
- Uses shared memory tiles of 16x16 pixels
- Loads halo regions (kernel radius) around each tile
- Computes box blur using shared memory data
- Reduces global memory accesses by ~90%

#### Gaussian Blur Kernel
- Uses shared memory for image tiles
- Stores kernel coefficients in constant memory
- Kernel size limited to 32x32 (MAX_KERNEL_SIZE)
- Normalized kernel copied to constant memory before launch

#### Sobel Edge Detection Kernel
- Uses shared memory with 1-pixel halo
- Sobel operators stored in shared memory (initialized once per block)
- Computes gradient magnitude using shared memory data

### Memory Management

#### PinnedBuffer
- Allocates page-locked host memory using `cudaHostAlloc()`
- Enables async transfers with `cudaMemcpyAsync()`
- Automatically freed in destructor

#### DeviceBuffer
- Manages device memory allocation
- Uses `cudaMalloc()` for allocation
- Automatically freed in destructor

#### StreamManager
- Manages multiple CUDA streams (default: 4 streams)
- Provides stream synchronization methods
- Enables pipelining of operations

### Tiling Strategy

For large images, the implementation can split processing into tiles:
- Tile size: 512x512 pixels (configurable)
- Overlap: kernel radius * 2 (to handle border effects)
- Multiple streams process different tiles concurrently
- While processing tile N, transfers for tile N+1 and N-1 occur asynchronously

## Performance Optimizations

1. **Shared Memory**: Reduces global memory bandwidth by caching frequently accessed data
2. **Constant Memory**: Fast access to filter coefficients (cached, read-only)
3. **Async Transfers**: Overlaps computation with data transfers
4. **Multiple Streams**: Enables concurrent kernel execution and transfers
5. **Pinned Memory**: Faster host-device transfers (up to 2x speedup)
6. **Minimal Transfers**: Keeps data on GPU between operations when possible

## Usage

The CUDA version provides the same interface as the CPU version, but with CUDA-accelerated operations for:
- Box Blur (option 30)
- Gaussian Blur (option 31)
- Median Filter (option 32)
- Bilateral Filter (option 33)
- Sobel Edge Detection (option 40)
- Canny Edge Detection (option 41)
- Sharpen (option 42)
- Prewitt (option 43)
- Laplacian (option 44)

All other operations (point operations, noise, morphological, geometric, color) use the CPU implementation.

## Notes

- The implementation is optimized for images that fit in GPU memory
- For very large images, consider using the tiled version
- Some operations (median filter, bilateral filter) are simplified in this implementation
- Full implementations would require additional kernel development

## Future Improvements

1. Complete median filter implementation using shared memory sorting
2. Full bilateral filter with optimized shared memory usage
3. Texture memory for read-only image data
4. Multi-GPU support for very large images
5. Dynamic tile sizing based on available shared memory

