#include "edge_detection_mpi.h"
#include "mpi_utils.h"
#include "edge_detection.h"
#include "point_operations.h"
#include <cmath>
#include <algorithm>
#include <queue>

namespace EdgeDetectionMPI {

// Helper function to apply Sobel edge detection to a specific row range
static void applySobelRegion(const Image& src, Image& dst, int startY, int endY) {
    const int width = src.getWidth();
    const int height = src.getHeight();
    
    // Convert to grayscale if needed
    Image gray = (src.getChannels() > 1) ? PointOps::grayscale(src) : src.clone();
    
    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0, gy = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = gray.getPixel(x + kx, y + ky);
                    gx += pixel[0] * kernelX[ky + 1][kx + 1];
                    gy += pixel[0] * kernelY[ky + 1][kx + 1];
                }
            }
            
            float magnitude = std::sqrt(gx * gx + gy * gy);
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, magnitude));
            dst.setPixel(x, y, &value);
        }
    }
}

Image sobel(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Convert to grayscale and create result
    Image gray = (exchange.extended.getChannels() > 1) ? 
                 PointOps::grayscale(exchange.extended) : exchange.extended.clone();
    Image filtered(gray.getWidth(), gray.getHeight(), 1);
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd && innerStart >= 1 && innerEnd <= gray.getHeight() - 1) {
        applySobelRegion(exchange.extended, filtered, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd && topStart >= 1) {
        applySobelRegion(exchange.extended, filtered, topStart, topEnd);
    }
    if (bottomStart < bottomEnd && bottomEnd <= gray.getHeight() - 1) {
        applySobelRegion(exchange.extended, filtered, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image canny(const Image& localChunk, double lowThreshold, double highThreshold, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // Canny uses Sobel (3x3 kernel, radius = 1)
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Note: Canny is complex with non-local operations (hysteresis),
    // so we process entire extended image after halo exchange
    // In production, you'd optimize by computing gradient/NMS on inner region first
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION: Apply Canny on full extended image ===
    // (Canny's hysteresis makes region-based computation complex)
    Image filtered = EdgeDetection::canny(exchange.extended, lowThreshold, highThreshold);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

// Helper function to apply sharpen filter to a specific row range
static void applySharpenRegion(const Image& src, Image& dst, int startY, int endY) {
    const int width = src.getWidth();
    const int height = src.getHeight();
    const int channels = src.getChannels();
    
    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            std::vector<float> sum(channels, 0.0f);
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = src.getPixel(x + kx, y + ky);
                    int weight = kernel[ky + 1][kx + 1];
                    for (int c = 0; c < channels; ++c) {
                        sum[c] += pixel[c] * weight;
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < channels; ++c) {
                pixel[c] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, sum[c])));
            }
            dst.setPixel(x, y, pixel);
        }
    }
}

Image sharpen(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Create result image
    Image filtered = exchange.extended.createSimilar();
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd && innerStart >= 1 && innerEnd <= exchange.extended.getHeight() - 1) {
        applySharpenRegion(exchange.extended, filtered, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd && topStart >= 1) {
        applySharpenRegion(exchange.extended, filtered, topStart, topEnd);
    }
    if (bottomStart < bottomEnd && bottomEnd <= exchange.extended.getHeight() - 1) {
        applySharpenRegion(exchange.extended, filtered, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

// Helper function to apply Prewitt edge detection to a specific row range
static void applyPrewittRegion(const Image& src, Image& dst, int startY, int endY) {
    const int width = src.getWidth();
    const int height = src.getHeight();
    
    // Convert to grayscale if needed
    Image gray = (src.getChannels() > 1) ? PointOps::grayscale(src) : src.clone();
    
    int kernelX[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = 0, gy = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = gray.getPixel(x + kx, y + ky);
                    gx += pixel[0] * kernelX[ky + 1][kx + 1];
                    gy += pixel[0] * kernelY[ky + 1][kx + 1];
                }
            }
            
            float magnitude = std::sqrt(gx * gx + gy * gy);
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, magnitude));
            dst.setPixel(x, y, &value);
        }
    }
}

Image prewitt(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Convert to grayscale and create result
    Image gray = (exchange.extended.getChannels() > 1) ? 
                 PointOps::grayscale(exchange.extended) : exchange.extended.clone();
    Image filtered(gray.getWidth(), gray.getHeight(), 1);
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd && innerStart >= 1 && innerEnd <= gray.getHeight() - 1) {
        applyPrewittRegion(exchange.extended, filtered, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd && topStart >= 1) {
        applyPrewittRegion(exchange.extended, filtered, topStart, topEnd);
    }
    if (bottomStart < bottomEnd && bottomEnd <= gray.getHeight() - 1) {
        applyPrewittRegion(exchange.extended, filtered, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

// Helper function to apply Laplacian edge detection to a specific row range
static void applyLaplacianRegion(const Image& src, Image& dst, int startY, int endY) {
    const int width = src.getWidth();
    const int height = src.getHeight();
    
    // Convert to grayscale if needed
    Image gray = (src.getChannels() > 1) ? PointOps::grayscale(src) : src.clone();
    
    int kernel[3][3] = {
        { 0,  1,  0},
        { 1, -4,  1},
        { 0,  1,  0}
    };
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sum = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = gray.getPixel(x + kx, y + ky);
                    sum += pixel[0] * kernel[ky + 1][kx + 1];
                }
            }
            
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, std::abs(sum)));
            dst.setPixel(x, y, &value);
        }
    }
}

Image laplacian(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Convert to grayscale and create result
    Image gray = (exchange.extended.getChannels() > 1) ? 
                 PointOps::grayscale(exchange.extended) : exchange.extended.clone();
    Image filtered(gray.getWidth(), gray.getHeight(), 1);
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd && innerStart >= 1 && innerEnd <= gray.getHeight() - 1) {
        applyLaplacianRegion(exchange.extended, filtered, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd && topStart >= 1) {
        applyLaplacianRegion(exchange.extended, filtered, topStart, topEnd);
    }
    if (bottomStart < bottomEnd && bottomEnd <= gray.getHeight() - 1) {
        applyLaplacianRegion(exchange.extended, filtered, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

} // namespace EdgeDetectionMPI
