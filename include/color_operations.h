#ifndef COLOR_OPERATIONS_H
#define COLOR_OPERATIONS_H

#include "image.h"
#include <vector>

namespace ColorOps {
    // Channel operations
    std::vector<Image> splitChannels(const Image& img);
    Image mergeChannels(const std::vector<Image>& channels);
    
    // Color space conversions
    Image rgbToHsv(const Image& img);
    Image hsvToRgb(const Image& img);
    
    // HSV adjustments
    Image adjustHue(const Image& img, float delta);  // -180 to 180
    Image adjustSaturation(const Image& img, float factor);  // 0.0 to 2.0
    Image adjustValue(const Image& img, float factor);  // 0.0 to 2.0
    
    // Advanced color operations (Bonus)
    Image rgbToLab(const Image& img);
    Image labToRgb(const Image& img);
    Image colorBalance(const Image& img, float redFactor, float greenFactor, float blueFactor);
    Image toneMapping(const Image& img, float exposure, float gamma);
}

#endif // COLOR_OPERATIONS_H
