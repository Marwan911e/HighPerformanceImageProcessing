#include "geometric.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace Geometric {

Image rotate(const Image& img, double angle, Interpolation interp) {
    if (!img.isValid()) return Image();
    
    double rad = angle * 3.14159265358979323846 / 180.0;
    double cosA = std::cos(rad);
    double sinA = std::sin(rad);
    
    int newWidth = static_cast<int>(std::abs(img.getWidth() * cosA) + std::abs(img.getHeight() * sinA));
    int newHeight = static_cast<int>(std::abs(img.getWidth() * sinA) + std::abs(img.getHeight() * cosA));
    
    Image result(newWidth, newHeight, img.getChannels());
    
    int cx = img.getWidth() / 2;
    int cy = img.getHeight() / 2;
    int ncx = newWidth / 2;
    int ncy = newHeight / 2;
    
    #pragma omp parallel for collapse(2) private(x, y, dx, dy, srcX, srcY, x0, y0, fx, fy, p00, p01, p10, p11, pixel, c, val) shared(img, result, cosA, sinA, cx, cy, ncx, ncy, interp)
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            int dx = x - ncx;
            int dy = y - ncy;
            
            double srcX = dx * cosA + dy * sinA + cx;
            double srcY = -dx * sinA + dy * cosA + cy;
            
            if (srcX >= 0 && srcX < img.getWidth() - 1 && srcY >= 0 && srcY < img.getHeight() - 1) {
                if (interp == Interpolation::NEAREST_NEIGHBOR) {
                    const uint8_t* pixel = img.getPixel(static_cast<int>(srcX + 0.5), static_cast<int>(srcY + 0.5));
                    result.setPixel(x, y, pixel);
                } else {
                    int x0 = static_cast<int>(srcX);
                    int y0 = static_cast<int>(srcY);
                    float fx = srcX - x0;
                    float fy = srcY - y0;
                    
                    const uint8_t* p00 = img.getPixel(x0, y0);
                    const uint8_t* p10 = img.getPixel(x0 + 1, y0);
                    const uint8_t* p01 = img.getPixel(x0, y0 + 1);
                    const uint8_t* p11 = img.getPixel(x0 + 1, y0 + 1);
                    
                    uint8_t pixel[4];
                    for (int c = 0; c < img.getChannels(); ++c) {
                        float val = p00[c] * (1 - fx) * (1 - fy) +
                                   p10[c] * fx * (1 - fy) +
                                   p01[c] * (1 - fx) * fy +
                                   p11[c] * fx * fy;
                        pixel[c] = static_cast<uint8_t>(val);
                    }
                    result.setPixel(x, y, pixel);
                }
            }
        }
    }
    
    return result;
}

Image scale(const Image& img, float scaleX, float scaleY, Interpolation interp) {
    if (!img.isValid() || scaleX <= 0 || scaleY <= 0) return Image();
    
    int newWidth = static_cast<int>(img.getWidth() * scaleX);
    int newHeight = static_cast<int>(img.getHeight() * scaleY);
    
    return resize(img, newWidth, newHeight, interp);
}

Image resize(const Image& img, int newWidth, int newHeight, Interpolation interp) {
    if (!img.isValid() || newWidth <= 0 || newHeight <= 0) return Image();
    
    Image result(newWidth, newHeight, img.getChannels());
    
    float scaleX = static_cast<float>(img.getWidth()) / newWidth;
    float scaleY = static_cast<float>(img.getHeight()) / newHeight;
    
    #pragma omp parallel for collapse(2) private(x, y, srcX, srcY, ix, iy, x0, y0, fx, fy, p00, p01, p10, p11, pixel, c, val) shared(img, result, scaleX, scaleY, interp)
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            float srcX = x * scaleX;
            float srcY = y * scaleY;
            
            if (interp == Interpolation::NEAREST_NEIGHBOR) {
                int ix = static_cast<int>(srcX + 0.5);
                int iy = static_cast<int>(srcY + 0.5);
                if (ix < img.getWidth() && iy < img.getHeight()) {
                    const uint8_t* pixel = img.getPixel(ix, iy);
                    result.setPixel(x, y, pixel);
                }
            } else {
                int x0 = static_cast<int>(srcX);
                int y0 = static_cast<int>(srcY);
                
                if (x0 < img.getWidth() - 1 && y0 < img.getHeight() - 1) {
                    float fx = srcX - x0;
                    float fy = srcY - y0;
                    
                    const uint8_t* p00 = img.getPixel(x0, y0);
                    const uint8_t* p10 = img.getPixel(x0 + 1, y0);
                    const uint8_t* p01 = img.getPixel(x0, y0 + 1);
                    const uint8_t* p11 = img.getPixel(x0 + 1, y0 + 1);
                    
                    uint8_t pixel[4];
                    for (int c = 0; c < img.getChannels(); ++c) {
                        float val = p00[c] * (1 - fx) * (1 - fy) +
                                   p10[c] * fx * (1 - fy) +
                                   p01[c] * (1 - fx) * fy +
                                   p11[c] * fx * fy;
                        pixel[c] = static_cast<uint8_t>(val);
                    }
                    result.setPixel(x, y, pixel);
                }
            }
        }
    }
    
    return result;
}

Image translate(const Image& img, int dx, int dy) {
    if (!img.isValid()) return Image();
    
    Image result = img.createSimilar();
    
    #pragma omp parallel for collapse(2) private(x, y, srcX, srcY, pixel) shared(img, result, dx, dy)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            int srcX = x - dx;
            int srcY = y - dy;
            
            if (srcX >= 0 && srcX < img.getWidth() && srcY >= 0 && srcY < img.getHeight()) {
                const uint8_t* pixel = img.getPixel(srcX, srcY);
                result.setPixel(x, y, pixel);
            }
        }
    }
    
    return result;
}

Image flipHorizontal(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image result = img.createSimilar();
    
    #pragma omp parallel for collapse(2) private(x, y, pixel) shared(img, result)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            const uint8_t* pixel = img.getPixel(img.getWidth() - 1 - x, y);
            result.setPixel(x, y, pixel);
        }
    }
    
    return result;
}

Image flipVertical(const Image& img) {
    if (!img.isValid()) return Image();
    
    Image result = img.createSimilar();
    
    #pragma omp parallel for collapse(2) private(x, y, pixel) shared(img, result)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            const uint8_t* pixel = img.getPixel(x, img.getHeight() - 1 - y);
            result.setPixel(x, y, pixel);
        }
    }
    
    return result;
}

Image perspectiveTransform(const Image& img, const float srcPoints[4][2], const float dstPoints[4][2]) {
    // Simplified perspective transform using homography matrix
    // This is a basic implementation - full implementation would solve for perspective matrix
    if (!img.isValid()) return Image();
    
    // For now, return a copy (full implementation requires matrix computation)
    return img.clone();
}

} // namespace Geometric
