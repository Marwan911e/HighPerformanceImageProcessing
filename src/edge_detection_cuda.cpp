#include "edge_detection_cuda.h"
#include "cuda_utils.h"
#include "cuda_kernels.cuh"
#include <cmath>
#include <cstring>

// Forward declaration
extern void launchSobelKernel(const uint8_t* d_input, uint8_t* d_output,
                             int width, int height, int channels,
                             cudaStream_t stream);

namespace EdgeDetectionCUDA {

Image sobel(const Image& img) {
    if (!img.isValid()) {
        return Image();
    }
    
    int width = img.getWidth();
    int height = img.getHeight();
    int channels = img.getChannels();
    size_t imageSize = img.getDataSize();
    
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
    
    // Launch Sobel kernel
    launchSobelKernel(d_input.getDevicePtr(), d_output.getDevicePtr(),
                     width, height, channels, stream);
    
    // Async D2H copy
    CUDA_CHECK(cudaMemcpyAsync(pinnedOutput.getHostPtr(), d_output.getDevicePtr(),
                               imageSize, cudaMemcpyDeviceToHost, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    std::memcpy(result.getData(), pinnedOutput.getHostPtr(), imageSize);
    
    return result;
}

Image canny(const Image& img, double lowThreshold, double highThreshold) {
    // Canny is complex - would need multiple kernel passes
    // Simplified version using Sobel as base
    if (!img.isValid()) {
        return Image();
    }
    
    // For now, use Sobel as placeholder
    // Full implementation would include:
    // 1. Gaussian blur
    // 2. Sobel gradient
    // 3. Non-maximum suppression
    // 4. Double threshold
    // 5. Edge tracking
    
    return sobel(img);
}

Image sharpen(const Image& img) {
    if (!img.isValid()) {
        return Image();
    }
    
    // Sharpen kernel: [0, -1, 0; -1, 5, -1; 0, -1, 0]
    // This can be implemented similar to Gaussian blur
    // For now, simplified version
    
    return img.clone();
}

Image prewitt(const Image& img) {
    // Similar to Sobel but with different kernels
    if (!img.isValid()) {
        return Image();
    }
    
    // Placeholder - would need separate kernel
    return sobel(img);
}

Image laplacian(const Image& img) {
    // Laplacian operator
    if (!img.isValid()) {
        return Image();
    }
    
    // Placeholder - would need separate kernel
    return img.clone();
}

} // namespace EdgeDetectionCUDA

