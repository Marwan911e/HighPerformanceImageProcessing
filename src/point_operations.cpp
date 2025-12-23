#include "point_operations.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace PointOps {

Image grayscale(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image result(img.getWidth(), img.getHeight(), 1);
    
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            const uint8_t* pixel = img.getPixel(x, y);
            uint8_t gray;
            
            if (img.getChannels() >= 3) {
                // Standard luminosity method
                gray = static_cast<uint8_t>(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]);
            } else {
                gray = pixel[0];
            }
            
            result.setPixel(x, y, &gray);
        }
    }
    
    return result;
}

Image adjustBrightness(const Image& img, int delta) {
    if (!img.isValid()) return Image();
    
    Image result = img.clone();
    uint8_t* data = result.getData();
    size_t size = result.getDataSize();
    
    for (size_t i = 0; i < size; ++i) {
        int val = data[i] + delta;
        data[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
    }
    
    return result;
}

Image adjustContrast(const Image& img, float factor) {
    if (!img.isValid()) return Image();
    
    Image result = img.clone();
    uint8_t* data = result.getData();
    size_t size = result.getDataSize();
    
    float F = (259.0f * (factor * 255.0f + 255.0f)) / (255.0f * (259.0f - factor * 255.0f));
    
    for (size_t i = 0; i < size; ++i) {
        float val = F * (data[i] - 128.0f) + 128.0f;
        data[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
    }
    
    return result;
}

Image threshold(const Image& img, uint8_t thresh) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? grayscale(img) : img.clone();
    uint8_t* data = gray.getData();
    size_t size = gray.getDataSize();
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = (data[i] > thresh) ? 255 : 0;
    }
    
    return gray;
}

Image thresholdOtsu(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? grayscale(img) : img.clone();
    
    // Calculate histogram
    int histogram[256] = {0};
    const uint8_t* data = gray.getData();
    size_t size = gray.getDataSize();
    
    for (size_t i = 0; i < size; ++i) {
        histogram[data[i]]++;
    }
    
    // Calculate Otsu threshold
    int total = gray.getWidth() * gray.getHeight();
    float sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += i * histogram[i];
    }
    
    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float maxVariance = 0;
    int optimalThreshold = 0;
    
    for (int t = 0; t < 256; ++t) {
        wB += histogram[t];
        if (wB == 0) continue;
        
        wF = total - wB;
        if (wF == 0) break;
        
        sumB += t * histogram[t];
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        
        float variance = wB * wF * (mB - mF) * (mB - mF);
        
        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = t;
        }
    }
    
    return threshold(gray, optimalThreshold);
}

Image adaptiveThreshold(const Image& img, int blockSize, int c, bool gaussian) {
    if (!img.isValid() || blockSize < 3 || blockSize % 2 == 0) return Image();
    
    Image gray = (img.getChannels() > 1) ? grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int radius = blockSize / 2;
    
    for (int y = 0; y < gray.getHeight(); ++y) {
        for (int x = 0; x < gray.getWidth(); ++x) {
            float sum = 0;
            float count = 0;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < gray.getWidth() && ny >= 0 && ny < gray.getHeight()) {
                        const uint8_t* pixel = gray.getPixel(nx, ny);
                        
                        if (gaussian) {
                            float weight = std::exp(-(dx*dx + dy*dy) / (2.0f * radius * radius));
                            sum += pixel[0] * weight;
                            count += weight;
                        } else {
                            sum += pixel[0];
                            count += 1;
                        }
                    }
                }
            }
            
            float mean = sum / count;
            const uint8_t* pixel = gray.getPixel(x, y);
            uint8_t value = (pixel[0] > mean - c) ? 255 : 0;
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

// Niblack adaptive thresholding (Bonus)
Image adaptiveThresholdNiblack(const Image& img, int windowSize, double k) {
    if (!img.isValid() || windowSize < 3 || windowSize % 2 == 0) return Image();
    
    Image gray = (img.getChannels() > 1) ? grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int radius = windowSize / 2;
    
    for (int y = 0; y < gray.getHeight(); ++y) {
        for (int x = 0; x < gray.getWidth(); ++x) {
            double sum = 0;
            double sumSq = 0;
            int count = 0;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < gray.getWidth() && ny >= 0 && ny < gray.getHeight()) {
                        const uint8_t* pixel = gray.getPixel(nx, ny);
                        sum += pixel[0];
                        sumSq += pixel[0] * pixel[0];
                        count++;
                    }
                }
            }
            
            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            double stddev = std::sqrt(variance);
            
            double thresh = mean + k * stddev;
            
            const uint8_t* pixel = gray.getPixel(x, y);
            uint8_t value = (pixel[0] > thresh) ? 255 : 0;
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

// Sauvola adaptive thresholding (Bonus)
Image adaptiveThresholdSauvola(const Image& img, int windowSize, double k, double r) {
    if (!img.isValid() || windowSize < 3 || windowSize % 2 == 0) return Image();
    
    Image gray = (img.getChannels() > 1) ? grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int radius = windowSize / 2;
    
    for (int y = 0; y < gray.getHeight(); ++y) {
        for (int x = 0; x < gray.getWidth(); ++x) {
            double sum = 0;
            double sumSq = 0;
            int count = 0;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < gray.getWidth() && ny >= 0 && ny < gray.getHeight()) {
                        const uint8_t* pixel = gray.getPixel(nx, ny);
                        sum += pixel[0];
                        sumSq += pixel[0] * pixel[0];
                        count++;
                    }
                }
            }
            
            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            double stddev = std::sqrt(variance);
            
            double thresh = mean * (1.0 + k * ((stddev / r) - 1.0));
            
            const uint8_t* pixel = gray.getPixel(x, y);
            uint8_t value = (pixel[0] > thresh) ? 255 : 0;
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image invert(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image result = img.clone();
    uint8_t* data = result.getData();
    size_t size = result.getDataSize();
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = 255 - data[i];
    }
    
    return result;
}

Image gammaCorrection(const Image& img, float gamma) {
    if (!img.isValid() || gamma <= 0) return Image();
    
    Image result = img.clone();
    uint8_t* data = result.getData();
    size_t size = result.getDataSize();
    
    // Build lookup table for efficiency
    uint8_t lut[256];
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uint8_t>(std::pow(i / 255.0f, gamma) * 255.0f);
    }
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = lut[data[i]];
    }
    
    return result;
}

} // namespace PointOps
