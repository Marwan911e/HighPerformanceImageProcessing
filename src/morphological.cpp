#include "morphological.h"
#include "point_operations.h"
#include <algorithm>
#include <queue>

namespace Morphological {

std::vector<std::vector<int>> getStructuringElement(StructuringElement shape, int size) {
    std::vector<std::vector<int>> element(size, std::vector<int>(size, 0));
    int center = size / 2;
    
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            switch (shape) {
                case StructuringElement::RECTANGLE:
                    element[y][x] = 1;
                    break;
                case StructuringElement::ELLIPSE:
                    {
                        int dx = x - center;
                        int dy = y - center;
                        if (dx * dx + dy * dy <= center * center) {
                            element[y][x] = 1;
                        }
                    }
                    break;
                case StructuringElement::CROSS:
                    if (x == center || y == center) {
                        element[y][x] = 1;
                    }
                    break;
            }
        }
    }
    
    return element;
}

Image erode(const Image& img, const std::vector<std::vector<int>>& kernel) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image result = gray.createSimilar();
    
    int kh = kernel.size();
    int kw = kernel[0].size();
    int cy = kh / 2;
    int cx = kw / 2;
    
    for (int y = 0; y < gray.getHeight(); ++y) {
        for (int x = 0; x < gray.getWidth(); ++x) {
            uint8_t minVal = 255;
            
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    if (kernel[ky][kx]) {
                        int nx = x + kx - cx;
                        int ny = y + ky - cy;
                        
                        if (nx >= 0 && nx < gray.getWidth() && ny >= 0 && ny < gray.getHeight()) {
                            const uint8_t* pixel = gray.getPixel(nx, ny);
                            minVal = std::min(minVal, pixel[0]);
                        }
                    }
                }
            }
            
            result.setPixel(x, y, &minVal);
        }
    }
    
    return result;
}

Image dilate(const Image& img, const std::vector<std::vector<int>>& kernel) {
    if (!img.isValid()) return Image();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    Image result = gray.createSimilar();
    
    int kh = kernel.size();
    int kw = kernel[0].size();
    int cy = kh / 2;
    int cx = kw / 2;
    
    for (int y = 0; y < gray.getHeight(); ++y) {
        for (int x = 0; x < gray.getWidth(); ++x) {
            uint8_t maxVal = 0;
            
            for (int ky = 0; ky < kh; ++ky) {
                for (int kx = 0; kx < kw; ++kx) {
                    if (kernel[ky][kx]) {
                        int nx = x + kx - cx;
                        int ny = y + ky - cy;
                        
                        if (nx >= 0 && nx < gray.getWidth() && ny >= 0 && ny < gray.getHeight()) {
                            const uint8_t* pixel = gray.getPixel(nx, ny);
                            maxVal = std::max(maxVal, pixel[0]);
                        }
                    }
                }
            }
            
            result.setPixel(x, y, &maxVal);
        }
    }
    
    return result;
}

Image opening(const Image& img, const std::vector<std::vector<int>>& kernel) {
    return dilate(erode(img, kernel), kernel);
}

Image closing(const Image& img, const std::vector<std::vector<int>>& kernel) {
    return erode(dilate(img, kernel), kernel);
}

Image morphologicalGradient(const Image& img, const std::vector<std::vector<int>>& kernel) {
    Image dilated = dilate(img, kernel);
    Image eroded = erode(img, kernel);
    Image result = dilated.createSimilar();
    
    for (int y = 0; y < result.getHeight(); ++y) {
        for (int x = 0; x < result.getWidth(); ++x) {
            const uint8_t* d = dilated.getPixel(x, y);
            const uint8_t* e = eroded.getPixel(x, y);
            uint8_t value = *d - *e;
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image topHat(const Image& img, const std::vector<std::vector<int>>& kernel) {
    Image opened = opening(img, kernel);
    Image result = img.createSimilar();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    
    for (int y = 0; y < result.getHeight(); ++y) {
        for (int x = 0; x < result.getWidth(); ++x) {
            const uint8_t* orig = gray.getPixel(x, y);
            const uint8_t* open = opened.getPixel(x, y);
            uint8_t value = *orig - *open;
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image blackHat(const Image& img, const std::vector<std::vector<int>>& kernel) {
    Image closed = closing(img, kernel);
    Image result = img.createSimilar();
    
    Image gray = (img.getChannels() > 1) ? PointOps::grayscale(img) : img.clone();
    
    for (int y = 0; y < result.getHeight(); ++y) {
        for (int x = 0; x < result.getWidth(); ++x) {
            const uint8_t* clos = closed.getPixel(x, y);
            const uint8_t* orig = gray.getPixel(x, y);
            uint8_t value = *clos - *orig;
            result.setPixel(x, y, &value);
        }
    }
    
    return result;
}

Image reconstruction(const Image& marker, const Image& mask) {
    if (!marker.isValid() || !mask.isValid()) return Image();
    if (marker.getWidth() != mask.getWidth() || marker.getHeight() != mask.getHeight()) {
        return Image();
    }
    
    Image result = marker.clone();
    bool changed = true;
    
    auto kernel = getStructuringElement(StructuringElement::RECTANGLE, 3);
    
    while (changed) {
        changed = false;
        Image dilated = dilate(result, kernel);
        
        for (int y = 0; y < result.getHeight(); ++y) {
            for (int x = 0; x < result.getWidth(); ++x) {
                uint8_t* res = result.getPixel(x, y);
                const uint8_t* dil = dilated.getPixel(x, y);
                const uint8_t* msk = mask.getPixel(x, y);
                
                uint8_t newVal = std::min(*dil, *msk);
                if (newVal != *res) {
                    *res = newVal;
                    changed = true;
                }
            }
        }
    }
    
    return result;
}

} // namespace Morphological
