#ifndef FILTERS_CUDA_H
#define FILTERS_CUDA_H

#include "image.h"

namespace FiltersCUDA {
    // Box blur with CUDA optimizations
    Image boxBlur(const Image& img, int kernelSize);
    
    // Gaussian blur with CUDA optimizations
    Image gaussianBlur(const Image& img, int kernelSize, float sigma);
    
    // Median filter with CUDA optimizations
    Image medianFilter(const Image& img, int kernelSize);
    
    // Bilateral filter with CUDA optimizations
    Image bilateralFilter(const Image& img, int diameter, 
                         double sigmaColor, double sigmaSpace);
}

#endif // FILTERS_CUDA_H

