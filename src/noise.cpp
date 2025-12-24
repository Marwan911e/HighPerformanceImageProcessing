#include "noise.h"
#include "point_operations.h"
#include <random>
#include <ctime>
#include <omp.h>

namespace Noise {

Image saltAndPepper(const Image& img, float amount) {
    if (!img.isValid() || amount < 0 || amount > 1) return Image();
    
    Image result = img.clone();
    std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int y = 0; y < result.getHeight(); ++y) {
        for (int x = 0; x < result.getWidth(); ++x) {
            float r = dist(rng);
            
            if (r < amount / 2) {
                // Salt (white)
                uint8_t white[4] = {255, 255, 255, 255};
                result.setPixel(x, y, white);
            } else if (r < amount) {
                // Pepper (black)
                uint8_t black[4] = {0, 0, 0, 0};
                result.setPixel(x, y, black);
            }
        }
    }
    
    return result;
}

Image gaussian(const Image& img, float mean, float stddev) {
    if (!img.isValid()) return Image();
    
    Image result = img.clone();
    uint8_t* data = result.getData();
    size_t size = result.getDataSize();
    
    #pragma omp parallel private(rng, dist, i, noise, val) shared(data, size, mean, stddev)
    {
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)) + omp_get_thread_num());
        std::normal_distribution<float> dist(mean, stddev);
        
        #pragma omp for
        for (size_t i = 0; i < size; ++i) {
            float noise = dist(rng);
            int val = static_cast<int>(data[i] + noise);
            data[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
        }
    }
    
    return result;
}

Image speckle(const Image& img, float variance) {
    if (!img.isValid()) return Image();
    
    Image result = img.clone();
    uint8_t* data = result.getData();
    size_t size = result.getDataSize();
    
    #pragma omp parallel private(rng, dist, i, noise, val) shared(data, size, variance)
    {
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)) + omp_get_thread_num());
        std::normal_distribution<float> dist(0.0f, std::sqrt(variance));
        
        #pragma omp for
        for (size_t i = 0; i < size; ++i) {
            float noise = 1.0f + dist(rng);
            int val = static_cast<int>(data[i] * noise);
            data[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
        }
    }
    
    return result;
}

} // namespace Noise
