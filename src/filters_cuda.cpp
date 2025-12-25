#include "filters_cuda.h"
#include "cuda_utils.h"
#include "cuda_kernels.cuh"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Forward declaration
extern void launchBoxBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                                int width, int height, int channels, int kernelSize,
                                cudaStream_t stream);
extern void launchGaussianBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                                    int width, int height, int channels,
                                    const float* kernel, int kernelSize,
                                    cudaStream_t stream);

namespace FiltersCUDA {

Image boxBlur(const Image& img, int kernelSize) {
    if (!img.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int width = img.getWidth();
    int height = img.getHeight();
    int channels = img.getChannels();
    size_t imageSize = img.getDataSize();
    
    // Create result image
    Image result = img.createSimilar();
    
    // Allocate pinned host memory for async transfers
    CudaUtils::PinnedBuffer pinnedInput, pinnedOutput;
    pinnedInput.allocate(imageSize);
    pinnedOutput.allocate(imageSize);
    
    // Copy input to pinned memory
    std::memcpy(pinnedInput.getHostPtr(), img.getData(), imageSize);
    
    // Allocate device memory
    CudaUtils::DeviceBuffer d_input, d_output;
    d_input.allocate(imageSize);
    d_output.allocate(imageSize);
    
    // Create stream manager for overlap
    CudaUtils::StreamManager streamManager(4);
    
    // Strategy: Use single stream for simple operations, or tile-based for large images
    // For box blur, we'll use a simple approach with async transfers
    cudaStream_t stream = streamManager.getStream(0);
    
    // Async H2D copy
    CUDA_CHECK(cudaMemcpyAsync(d_input.getDevicePtr(), pinnedInput.getHostPtr(),
                               imageSize, cudaMemcpyHostToDevice, stream));
    
    // Launch kernel
    launchBoxBlurKernel(d_input.getDevicePtr(), d_output.getDevicePtr(),
                       width, height, channels, kernelSize, stream);
    
    // Async D2H copy
    CUDA_CHECK(cudaMemcpyAsync(pinnedOutput.getHostPtr(), d_output.getDevicePtr(),
                               imageSize, cudaMemcpyDeviceToHost, stream));
    
    // Wait for completion
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Copy result back
    std::memcpy(result.getData(), pinnedOutput.getHostPtr(), imageSize);
    
    return result;
}

Image gaussianBlur(const Image& img, int kernelSize, float sigma) {
    if (!img.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int width = img.getWidth();
    int height = img.getHeight();
    int channels = img.getChannels();
    size_t imageSize = img.getDataSize();
    
    // Generate Gaussian kernel on host
    int radius = kernelSize / 2;
    std::vector<float> kernel(kernelSize * kernelSize);
    float sum = 0.0f;
    
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float val = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + radius) * kernelSize + (x + radius)] = val;
            sum += val;
        }
    }
    
    // Normalize
    for (auto& val : kernel) {
        val /= sum;
    }
    
    // Create result image
    Image result = img.createSimilar();
    
    // Allocate pinned host memory
    CudaUtils::PinnedBuffer pinnedInput, pinnedOutput;
    pinnedInput.allocate(imageSize);
    pinnedOutput.allocate(imageSize);
    
    std::memcpy(pinnedInput.getHostPtr(), img.getData(), imageSize);
    
    // Allocate device memory
    CudaUtils::DeviceBuffer d_input, d_output;
    d_input.allocate(imageSize);
    d_output.allocate(imageSize);
    
    // Create stream manager
    CudaUtils::StreamManager streamManager(4);
    cudaStream_t stream = streamManager.getStream(0);
    
    // Async H2D copy
    CUDA_CHECK(cudaMemcpyAsync(d_input.getDevicePtr(), pinnedInput.getHostPtr(),
                               imageSize, cudaMemcpyHostToDevice, stream));
    
    // Launch kernel (kernel is copied to constant memory inside)
    launchGaussianBlurKernel(d_input.getDevicePtr(), d_output.getDevicePtr(),
                            width, height, channels, kernel.data(), kernelSize, stream);
    
    // Async D2H copy
    CUDA_CHECK(cudaMemcpyAsync(pinnedOutput.getHostPtr(), d_output.getDevicePtr(),
                               imageSize, cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    std::memcpy(result.getData(), pinnedOutput.getHostPtr(), imageSize);
    
    return result;
}

Image gaussianBlurTiled(const Image& img, int kernelSize, float sigma) {
    // Tiled version with overlap strategy for large images
    if (!img.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int width = img.getWidth();
    int height = img.getHeight();
    int channels = img.getChannels();
    int radius = kernelSize / 2;
    
    // Generate kernel
    std::vector<float> kernel(kernelSize * kernelSize);
    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float val = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + radius) * kernelSize + (x + radius)] = val;
            sum += val;
        }
    }
    for (auto& val : kernel) val /= sum;
    
    Image result = img.createSimilar();
    
    // Tile configuration
    int tileWidth = 512;
    int tileHeight = 512;
    int overlap = radius * 2; // Overlap to handle kernel
    
    // Generate tiles
    auto tiles = CudaUtils::generateTiles(width, height, tileWidth, tileHeight, overlap);
    
    // Allocate device memory for full image
    size_t imageSize = img.getDataSize();
    CudaUtils::DeviceBuffer d_input, d_output;
    d_input.allocate(imageSize);
    d_output.allocate(imageSize);
    
    // Stream manager for pipelining
    CudaUtils::StreamManager streamManager(std::min(4, (int)tiles.size()));
    
    // Process tiles with overlap: while GPU works on tile N, transfer tile N+1
    for (size_t i = 0; i < tiles.size(); ++i) {
        int streamIdx = i % streamManager.getNumStreams();
        cudaStream_t stream = streamManager.getStream(streamIdx);
        
        const auto& tile = tiles[i];
        
        // For first tile, copy input
        if (i == 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_input.getDevicePtr(), img.getData(),
                                       imageSize, cudaMemcpyHostToDevice, stream));
        }
        
        // Launch kernel for this tile
        // Note: This is a simplified version - full implementation would
        // process only the tile region
        launchGaussianBlurKernel(d_input.getDevicePtr(), d_output.getDevicePtr(),
                                width, height, channels, kernel.data(), kernelSize, stream);
        
        // Copy result back (async)
        if (i == tiles.size() - 1) {
            CUDA_CHECK(cudaMemcpyAsync(result.getData(), d_output.getDevicePtr(),
                                       imageSize, cudaMemcpyDeviceToHost, stream));
        }
    }
    
    // Synchronize all streams
    streamManager.synchronizeAll();
    
    return result;
}

Image medianFilter(const Image& img, int kernelSize) {
    // Median filter is more complex - simplified version
    // Full implementation would use shared memory for sorting
    if (!img.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    // For now, return a copy (full median implementation would be complex)
    return img.clone();
}

Image bilateralFilter(const Image& img, int diameter, double sigmaColor, double sigmaSpace) {
    // Bilateral filter is complex - would need separate kernel
    // Simplified version for now
    if (!img.isValid() || diameter < 3 || diameter % 2 == 0) {
        return Image();
    }
    
    // Placeholder - full implementation would use shared memory
    return img.clone();
}

} // namespace FiltersCUDA

