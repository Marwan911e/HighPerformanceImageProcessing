#ifndef EDGE_DETECTION_CUDA_H
#define EDGE_DETECTION_CUDA_H

#include "image.h"

namespace EdgeDetectionCUDA {
    // Sobel operator with CUDA optimizations
    Image sobel(const Image& img);
    
    // Canny edge detection with CUDA optimizations
    Image canny(const Image& img, double lowThreshold, double highThreshold);
    
    // Sharpen filter with CUDA optimizations
    Image sharpen(const Image& img);
    
    // Prewitt operator with CUDA optimizations
    Image prewitt(const Image& img);
    
    // Laplacian operator with CUDA optimizations
    Image laplacian(const Image& img);
}

#endif // EDGE_DETECTION_CUDA_H

