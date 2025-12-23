#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include "image.h"

namespace EdgeDetection {
    // Sobel operator
    Image sobel(const Image& img);
    Image sobelX(const Image& img);
    Image sobelY(const Image& img);
    
    // Canny edge detection
    Image canny(const Image& img, double lowThreshold, double highThreshold);
    
    // Sharpen filter
    Image sharpen(const Image& img);
    
    // Prewitt operator (Bonus)
    Image prewitt(const Image& img);
    
    // Laplacian operator (Bonus)
    Image laplacian(const Image& img);
}

#endif // EDGE_DETECTION_H
