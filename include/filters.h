#ifndef FILTERS_H
#define FILTERS_H

#include "image.h"

namespace Filters {
    // Blur operations
    Image boxBlur(const Image& img, int kernelSize);
    Image gaussianBlur(const Image& img, int kernelSize, float sigma);
    Image medianFilter(const Image& img, int kernelSize);
    
    // Bilateral filter (Bonus)
    Image bilateralFilter(const Image& img, int diameter, double sigmaColor, double sigmaSpace);
}

#endif // FILTERS_H
