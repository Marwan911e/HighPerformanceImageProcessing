#include "edge_detection.h"
#include "point_operations.h"
#include "filters.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <omp.h>

namespace EdgeDetection {

Image sobelX(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int kernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    #pragma omp parallel for collapse(2) private(x, y, gx, ky, kx, pixel, value) shared(gray, result, kernelX)
    for (int y = 1; y < gray.getHeight() - 1; ++y) {
        for (int x = 1; x < gray.getWidth() - 1; ++x) {
            float gx = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = gray.getPixel(x + kx, y + ky);
                    gx += pixel[0] * kernelX[ky + 1][kx + 1];
                }
            }
            
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, std::abs(gx)));
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image sobelY(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int kernelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    #pragma omp parallel for collapse(2) private(x, y, gy, ky, kx, pixel, value) shared(gray, result, kernelY)
    for (int y = 1; y < gray.getHeight() - 1; ++y) {
        for (int x = 1; x < gray.getWidth() - 1; ++x) {
            float gy = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = gray.getPixel(x + kx, y + ky);
                    gy += pixel[0] * kernelY[ky + 1][kx + 1];
                }
            }
            
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, std::abs(gy)));
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image sobel(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    #pragma omp parallel for collapse(2) private(x, y, gx, gy, ky, kx, pixel, magnitude, value) shared(gray, result, kernelX, kernelY)
    for (int y = 1; y < gray.getHeight() - 1; ++y) {
        for (int x = 1; x < gray.getWidth() - 1; ++x) {
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
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image canny(const Image& img, double lowThreshold, double highThreshold) {
    if (!img.isValid()) return Image();
    
    // Step 1: Convert to grayscale and apply Gaussian blur
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image blurred = Filters::gaussianBlur(gray, 5, 1.4f);
    
    // Step 2: Calculate gradients
    Image gradMag(blurred.getWidth(), blurred.getHeight(), 1);
    Image gradDir(blurred.getWidth(), blurred.getHeight(), 1);
    
    int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    #pragma omp parallel for collapse(2) private(x, y, gx, gy, ky, kx, pixel, magnitude, mag, angle, dir) shared(blurred, gradMag, gradDir, kernelX, kernelY)
    for (int y = 1; y < blurred.getHeight() - 1; ++y) {
        for (int x = 1; x < blurred.getWidth() - 1; ++x) {
            float gx = 0, gy = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = blurred.getPixel(x + kx, y + ky);
                    gx += pixel[0] * kernelX[ky + 1][kx + 1];
                    gy += pixel[0] * kernelY[ky + 1][kx + 1];
                }
            }
            
            float magnitude = std::sqrt(gx * gx + gy * gy);
            uint8_t mag = static_cast<uint8_t>(std::min(255.0f, magnitude));
            gradMag.setPixel(x, y, &mag);
            
            float angle = std::atan2(gy, gx) * 180.0f / 3.14159f;
            uint8_t dir = static_cast<uint8_t>((angle + 180.0f) / 45.0f);
            gradDir.setPixel(x, y, &dir);
        }
    }
    
    // Step 3: Non-maximum suppression
    Image suppressed(gradMag.getWidth(), gradMag.getHeight(), 1);
    
    #pragma omp parallel for collapse(2) private(x, y, mag, dir, neighbor1, neighbor2, value) shared(gradMag, gradDir, suppressed)
    for (int y = 1; y < gradMag.getHeight() - 1; ++y) {
        for (int x = 1; x < gradMag.getWidth() - 1; ++x) {
            const uint8_t* mag = gradMag.getPixel(x, y);
            const uint8_t* dir = gradDir.getPixel(x, y);
            
            uint8_t neighbor1 = 0, neighbor2 = 0;
            
            switch (*dir % 4) {
                case 0: // Horizontal
                    neighbor1 = *gradMag.getPixel(x - 1, y);
                    neighbor2 = *gradMag.getPixel(x + 1, y);
                    break;
                case 1: // Diagonal /
                    neighbor1 = *gradMag.getPixel(x - 1, y + 1);
                    neighbor2 = *gradMag.getPixel(x + 1, y - 1);
                    break;
                case 2: // Vertical
                    neighbor1 = *gradMag.getPixel(x, y - 1);
                    neighbor2 = *gradMag.getPixel(x, y + 1);
                    break;
                case 3: // Diagonal backslash
                    neighbor1 = *gradMag.getPixel(x - 1, y - 1);
                    neighbor2 = *gradMag.getPixel(x + 1, y + 1);
                    break;
            }
            
            uint8_t value = (*mag >= neighbor1 && *mag >= neighbor2) ? *mag : 0;
            suppressed.setPixel(x, y, &value);
        }
    }
    
    // Step 4: Double thresholding and edge tracking by hysteresis
    Image result(suppressed.getWidth(), suppressed.getHeight(), 1);
    
    #pragma omp parallel for collapse(2) private(x, y, pixel, value) shared(suppressed, result, lowThreshold, highThreshold)
    for (int y = 0; y < suppressed.getHeight(); ++y) {
        for (int x = 0; x < suppressed.getWidth(); ++x) {
            const uint8_t* pixel = suppressed.getPixel(x, y);
            uint8_t value;
            
            if (*pixel >= highThreshold) {
                value = 255; // Strong edge
            } else if (*pixel >= lowThreshold) {
                value = 128; // Weak edge
            } else {
                value = 0; // Non-edge
            }
            
            result.setPixel(x, y, &value);
        }
    }
    
    // Edge tracking by hysteresis
    for (int y = 1; y < result.getHeight() - 1; ++y) {
        for (int x = 1; x < result.getWidth() - 1; ++x) {
            uint8_t* pixel = result.getPixel(x, y);
            
            if (*pixel == 128) {
                bool connected = false;
                for (int dy = -1; dy <= 1 && !connected; ++dy) {
                    for (int dx = -1; dx <= 1 && !connected; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const uint8_t* neighbor = result.getPixel(x + dx, y + dy);
                        if (*neighbor == 255) {
                            connected = true;
                        }
                    }
                }
                *pixel = connected ? 255 : 0;
            }
        }
    }
    
    return result;
}

Image sharpen(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image result = img.createSimilar();
    
    float kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    
    #pragma omp parallel for collapse(2) private(x, y, accum, ky, kx, pixel, c) shared(img, result, kernel)
    for (int y = 1; y < img.getHeight() - 1; ++y) {
        for (int x = 1; x < img.getWidth() - 1; ++x) {
            std::vector<float> accum(img.getChannels(), 0.0f);
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = img.getPixel(x + kx, y + ky);
                    for (int c = 0; c < img.getChannels(); ++c) {
                        accum[c] += pixel[c] * kernel[ky + 1][kx + 1];
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

Image prewitt(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int kernelX[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    int kernelY[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    
    #pragma omp parallel for collapse(2) private(x, y, gx, gy, ky, kx, pixel, magnitude, value) shared(gray, result, kernelX, kernelY)
    for (int y = 1; y < gray.getHeight() - 1; ++y) {
        for (int x = 1; x < gray.getWidth() - 1; ++x) {
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
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image laplacian(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image result(gray.getWidth(), gray.getHeight(), 1);
    
    int kernel[3][3] = {
        { 0,  1,  0},
        { 1, -4,  1},
        { 0,  1,  0}
    };
    
    #pragma omp parallel for collapse(2) private(x, y, sum, ky, kx, pixel, value) shared(gray, result, kernel)
    for (int y = 1; y < gray.getHeight() - 1; ++y) {
        for (int x = 1; x < gray.getWidth() - 1; ++x) {
            float sum = 0;
            
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const uint8_t* pixel = gray.getPixel(x + kx, y + ky);
                    sum += pixel[0] * kernel[ky + 1][kx + 1];
                }
            }
            
            uint8_t value = static_cast<uint8_t>(std::min(255.0f, std::abs(sum)));
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

} // namespace EdgeDetection
