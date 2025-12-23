#ifndef MORPHOLOGICAL_H
#define MORPHOLOGICAL_H

#include "image.h"
#include <vector>

namespace Morphological {
    // Structuring element types
    enum class StructuringElement {
        RECTANGLE,
        ELLIPSE,
        CROSS
    };
    
    // Generate structuring element
    std::vector<std::vector<int>> getStructuringElement(StructuringElement shape, int size);
    
    // Basic operations
    Image erode(const Image& img, const std::vector<std::vector<int>>& kernel);
    Image dilate(const Image& img, const std::vector<std::vector<int>>& kernel);
    
    // Combined operations
    Image opening(const Image& img, const std::vector<std::vector<int>>& kernel);
    Image closing(const Image& img, const std::vector<std::vector<int>>& kernel);
    
    // Advanced operations
    Image morphologicalGradient(const Image& img, const std::vector<std::vector<int>>& kernel);
    Image topHat(const Image& img, const std::vector<std::vector<int>>& kernel);
    Image blackHat(const Image& img, const std::vector<std::vector<int>>& kernel);
    
    // Morphological reconstruction (Bonus)
    Image reconstruction(const Image& marker, const Image& mask);
}

#endif // MORPHOLOGICAL_H
