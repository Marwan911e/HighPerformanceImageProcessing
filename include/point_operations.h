#ifndef POINT_OPERATIONS_H
#define POINT_OPERATIONS_H

#include "image.h"

namespace PointOps {
    // Basic conversions
    Image grayscale(const Image& img);
    Image adjustBrightness(const Image& img, int delta);  // -255 to 255
    Image adjustContrast(const Image& img, float factor);  // 0.0 to 3.0
    
    // Thresholding
    Image threshold(const Image& img, uint8_t thresh);
    Image thresholdOtsu(const Image& img);  // Automatic Otsu thresholding
    Image adaptiveThreshold(const Image& img, int blockSize, int c, bool gaussian = true);
    
    // Advanced thresholding (Bonus)
    Image adaptiveThresholdNiblack(const Image& img, int windowSize, double k);
    Image adaptiveThresholdSauvola(const Image& img, int windowSize, double k, double r);
    
    // Transformations
    Image invert(const Image& img);
    Image gammaCorrection(const Image& img, float gamma);
}

#endif // POINT_OPERATIONS_H
