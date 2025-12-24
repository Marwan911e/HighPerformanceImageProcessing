#include "filters.h"
#include "point_operations.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>

namespace Filters {

Image boxBlur(const Image& img, int kernelSize) {
    if (!img.isValid() || kernelSize < 3 || kernelSize % 2 == 0) return Image();
    
    Image result = img.createSimilar();
    int radius = kernelSize / 2;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            std::vector<float> sum(img.getChannels(), 0.0f);
            int count = 0;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < img.getWidth() && ny >= 0 && ny < img.getHeight()) {
                        const uint8_t* pixel = img.getPixel(nx, ny);
                        for (int c = 0; c < img.getChannels(); ++c) {
                            sum[c] += pixel[c];
                        }
                        count++;
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < img.getChannels(); ++c) {
                pixel[c] = static_cast<uint8_t>(sum[c] / count);
            }
            result.setPixel(x, y, pixel);
        }
    }
    
    return result;
}

Image gaussianBlur(const Image& img, int kernelSize, float sigma) {
    if (!img.isValid() || kernelSize < 3 || kernelSize % 2 == 0) return Image();
    
    // Generate Gaussian kernel
    int radius = kernelSize / 2;
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    float sum = 0.0f;
    
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float val = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[y + radius][x + radius] = val;
            sum += val;
        }
    }
    
    // Normalize kernel
    for (int y = 0; y < kernelSize; ++y) {
        for (int x = 0; x < kernelSize; ++x) {
            kernel[y][x] /= sum;
        }
    }
    
    // Apply kernel
    Image result = img.createSimilar();
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            std::vector<float> accum(img.getChannels(), 0.0f);
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < img.getWidth() && ny >= 0 && ny < img.getHeight()) {
                        const uint8_t* pixel = img.getPixel(nx, ny);
                        float weight = kernel[dy + radius][dx + radius];
                        
                        for (int c = 0; c < img.getChannels(); ++c) {
                            accum[c] += pixel[c] * weight;
                        }
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < img.getChannels(); ++c) {
                pixel[c] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, accum[c])));
            }
            result.setPixel(x, y, pixel);
        }
    }
    
    return result;
}

Image medianFilter(const Image& img, int kernelSize) {
    if (!img.isValid() || kernelSize < 3 || kernelSize % 2 == 0) return Image();
    
    Image result = img.createSimilar();
    int radius = kernelSize / 2;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            std::vector<std::vector<uint8_t>> values(img.getChannels());
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < img.getWidth() && ny >= 0 && ny < img.getHeight()) {
                        const uint8_t* pixel = img.getPixel(nx, ny);
                        for (int c = 0; c < img.getChannels(); ++c) {
                            values[c].push_back(pixel[c]);
                        }
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < img.getChannels(); ++c) {
                std::nth_element(values[c].begin(), 
                                values[c].begin() + values[c].size() / 2, 
                                values[c].end());
                pixel[c] = values[c][values[c].size() / 2];
            }
            result.setPixel(x, y, pixel);
        }
    }
    
    return result;
}

Image bilateralFilter(const Image& img, int diameter, double sigmaColor, double sigmaSpace) {
    if (!img.isValid() || diameter < 3 || diameter % 2 == 0) return Image();
    
    Image result = img.createSimilar();
    int radius = diameter / 2;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            const uint8_t* centerPixel = img.getPixel(x, y);
            std::vector<float> accum(img.getChannels(), 0.0f);
            float totalWeight = 0.0f;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < img.getWidth() && ny >= 0 && ny < img.getHeight()) {
                        const uint8_t* pixel = img.getPixel(nx, ny);
                        
                        // Spatial weight
                        float spatialDist = std::sqrt(dx * dx + dy * dy);
                        float spatialWeight = std::exp(-(spatialDist * spatialDist) / (2 * sigmaSpace * sigmaSpace));
                        
                        // Color weight
                        float colorDist = 0;
                        for (int c = 0; c < img.getChannels(); ++c) {
                            float diff = centerPixel[c] - pixel[c];
                            colorDist += diff * diff;
                        }
                        colorDist = std::sqrt(colorDist);
                        float colorWeight = std::exp(-(colorDist * colorDist) / (2 * sigmaColor * sigmaColor));
                        
                        float weight = spatialWeight * colorWeight;
                        
                        for (int c = 0; c < img.getChannels(); ++c) {
                            accum[c] += pixel[c] * weight;
                        }
                        totalWeight += weight;
                    }
                }
            }
            
            uint8_t pixel[4];
            for (int c = 0; c < img.getChannels(); ++c) {
                pixel[c] = static_cast<uint8_t>(accum[c] / totalWeight);
            }
            result.setPixel(x, y, pixel);
        }
    }
    
    return result;
}

} // namespace Filters
