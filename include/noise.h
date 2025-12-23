#ifndef NOISE_H
#define NOISE_H

#include "image.h"

namespace Noise {
    // Add salt and pepper noise
    Image saltAndPepper(const Image& img, float amount);
    
    // Add Gaussian noise
    Image gaussian(const Image& img, float mean, float stddev);
    
    // Add speckle noise (Bonus)
    Image speckle(const Image& img, float variance);
}

#endif // NOISE_H
