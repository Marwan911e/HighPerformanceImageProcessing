#include "filters_mpi.h"
#include "mpi_utils.h"
#include "filters.h"
#include <algorithm>
#include <vector>
#include <cmath>

namespace FiltersMPI {

// Helper function to apply box blur to a specific row range
static void applyBoxBlurRegion(const Image& src, Image& dst, int kernelSize, 
                               int startY, int endY) {
    const int radius = kernelSize / 2;
    const int width = src.getWidth();
    const int height = src.getHeight();
    const int channels = src.getChannels();
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<float> sum(channels, 0.0f);
            int count = 0;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const uint8_t* pixel = src.getPixel(nx, ny);
                        for (int c = 0; c < channels; ++c) {
                            sum[c] += pixel[c];
                        }
                        count++;
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < channels; ++c) {
                pixel[c] = static_cast<uint8_t>(sum[c] / count);
            }
            dst.setPixel(x, y, pixel);
        }
    }
}

Image boxBlur(const Image& localChunk, int kernelSize, int rank, int size) {
    if (!localChunk.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int radius = kernelSize / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Create result image
    Image filtered = exchange.extended.createSimilar();
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd) {
        applyBoxBlurRegion(exchange.extended, filtered, kernelSize, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd) {
        applyBoxBlurRegion(exchange.extended, filtered, kernelSize, topStart, topEnd);
    }
    if (bottomStart < bottomEnd) {
        applyBoxBlurRegion(exchange.extended, filtered, kernelSize, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

// Helper function to apply gaussian blur to a specific row range
static void applyGaussianBlurRegion(const Image& src, Image& dst, 
                                    const std::vector<std::vector<float>>& kernel,
                                    int kernelSize, int startY, int endY) {
    const int radius = kernelSize / 2;
    const int width = src.getWidth();
    const int height = src.getHeight();
    const int channels = src.getChannels();
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<float> sum(channels, 0.0f);
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const uint8_t* pixel = src.getPixel(nx, ny);
                        float weight = kernel[dy + radius][dx + radius];
                        for (int c = 0; c < channels; ++c) {
                            sum[c] += pixel[c] * weight;
                        }
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

Image gaussianBlur(const Image& localChunk, int kernelSize, float sigma, int rank, int size) {
    if (!localChunk.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int radius = kernelSize / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // Generate Gaussian kernel (small computation, done once)
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    float sum = 0.0f;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float val = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[y + radius][x + radius] = val;
            sum += val;
        }
    }
    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            kernel[y][x] /= sum;
        }
    }
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Create result image
    Image filtered = exchange.extended.createSimilar();
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd) {
        applyGaussianBlurRegion(exchange.extended, filtered, kernel, kernelSize, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd) {
        applyGaussianBlurRegion(exchange.extended, filtered, kernel, kernelSize, topStart, topEnd);
    }
    if (bottomStart < bottomEnd) {
        applyGaussianBlurRegion(exchange.extended, filtered, kernel, kernelSize, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

// Helper function to apply median filter to a specific row range
static void applyMedianFilterRegion(const Image& src, Image& dst, int kernelSize,
                                    int startY, int endY) {
    const int radius = kernelSize / 2;
    const int width = src.getWidth();
    const int height = src.getHeight();
    const int channels = src.getChannels();
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<std::vector<uint8_t>> values(channels);
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const uint8_t* pixel = src.getPixel(nx, ny);
                        for (int c = 0; c < channels; ++c) {
                            values[c].push_back(pixel[c]);
                        }
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < channels; ++c) {
                std::sort(values[c].begin(), values[c].end());
                pixel[c] = values[c][values[c].size() / 2];
            }
            dst.setPixel(x, y, pixel);
        }
    }
}

Image medianFilter(const Image& localChunk, int kernelSize, int rank, int size) {
    if (!localChunk.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int radius = kernelSize / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Create result image
    Image filtered = exchange.extended.createSimilar();
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd) {
        applyMedianFilterRegion(exchange.extended, filtered, kernelSize, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd) {
        applyMedianFilterRegion(exchange.extended, filtered, kernelSize, topStart, topEnd);
    }
    if (bottomStart < bottomEnd) {
        applyMedianFilterRegion(exchange.extended, filtered, kernelSize, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

// Helper function to apply bilateral filter to a specific row range
static void applyBilateralFilterRegion(const Image& src, Image& dst, int diameter,
                                       double sigmaColor, double sigmaSpace,
                                       int startY, int endY) {
    const int radius = diameter / 2;
    const int width = src.getWidth();
    const int height = src.getHeight();
    const int channels = src.getChannels();
    
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<double> sum(channels, 0.0);
            double weightSum = 0.0;
            const uint8_t* centerPixel = src.getPixel(x, y);
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        const uint8_t* pixel = src.getPixel(nx, ny);
                        
                        // Spatial weight
                        double spatialDist = dx * dx + dy * dy;
                        double spatialWeight = std::exp(-spatialDist / (2.0 * sigmaSpace * sigmaSpace));
                        
                        // Color weight
                        double colorDist = 0.0;
                        for (int c = 0; c < channels; ++c) {
                            double diff = centerPixel[c] - pixel[c];
                            colorDist += diff * diff;
                        }
                        double colorWeight = std::exp(-colorDist / (2.0 * sigmaColor * sigmaColor));
                        
                        double weight = spatialWeight * colorWeight;
                        weightSum += weight;
                        
                        for (int c = 0; c < channels; ++c) {
                            sum[c] += pixel[c] * weight;
                        }
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < channels; ++c) {
                pixel[c] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, sum[c] / weightSum)));
            }
            dst.setPixel(x, y, pixel);
        }
    }
}

Image bilateralFilter(const Image& localChunk, int diameter, double sigmaColor, double sigmaSpace, int rank, int size) {
    if (!localChunk.isValid() || diameter < 3 || diameter % 2 == 0) {
        return Image();
    }
    
    int radius = diameter / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // === OVERLAP PATTERN: Start non-blocking communication ===
    MPIUtils::BoundaryExchange exchange = MPIUtils::startBoundaryExchange(localChunk, rank, size, haloSize);
    
    // Create result image
    Image filtered = exchange.extended.createSimilar();
    
    // === COMPUTATION PHASE 1: Compute inner region while waiting for halos ===
    int innerStart, innerEnd;
    MPIUtils::getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd) {
        applyBilateralFilterRegion(exchange.extended, filtered, diameter, sigmaColor, sigmaSpace, innerStart, innerEnd);
    }
    
    // === WAIT: Ensure halo data has arrived ===
    MPIUtils::waitBoundaryExchange(exchange);
    
    // === COMPUTATION PHASE 2: Compute border regions (need halo data) ===
    int topStart, topEnd, bottomStart, bottomEnd;
    MPIUtils::getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                               topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd) {
        applyBilateralFilterRegion(exchange.extended, filtered, diameter, sigmaColor, sigmaSpace, topStart, topEnd);
    }
    if (bottomStart < bottomEnd) {
        applyBilateralFilterRegion(exchange.extended, filtered, diameter, sigmaColor, sigmaSpace, bottomStart, bottomEnd);
    }
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

} // namespace FiltersMPI
