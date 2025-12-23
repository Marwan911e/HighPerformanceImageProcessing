#ifndef GEOMETRIC_H
#define GEOMETRIC_H

#include "image.h"

namespace Geometric {
    // Interpolation methods
    enum class Interpolation {
        NEAREST_NEIGHBOR,
        BILINEAR,
        BICUBIC
    };
    
    // Basic transformations
    Image rotate(const Image& img, double angle, Interpolation interp = Interpolation::BILINEAR);
    Image scale(const Image& img, float scaleX, float scaleY, Interpolation interp = Interpolation::BILINEAR);
    Image resize(const Image& img, int newWidth, int newHeight, Interpolation interp = Interpolation::BILINEAR);
    Image translate(const Image& img, int dx, int dy);
    
    // Flipping
    Image flipHorizontal(const Image& img);
    Image flipVertical(const Image& img);
    
    // Perspective transform (Bonus)
    Image perspectiveTransform(const Image& img, const float srcPoints[4][2], const float dstPoints[4][2]);
}

#endif // GEOMETRIC_H
