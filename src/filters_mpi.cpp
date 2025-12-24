#include "filters_mpi.h"
#include "mpi_utils.h"
#include "filters.h"
#include <algorithm>

namespace FiltersMPI {

Image boxBlur(const Image& localChunk, int kernelSize, int rank, int size) {
    if (!localChunk.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int radius = kernelSize / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image (now has halo rows)
    Image filtered = Filters::boxBlur(extended, kernelSize);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image gaussianBlur(const Image& localChunk, int kernelSize, float sigma, int rank, int size) {
    if (!localChunk.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int radius = kernelSize / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image (now has halo rows)
    Image filtered = Filters::gaussianBlur(extended, kernelSize, sigma);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image medianFilter(const Image& localChunk, int kernelSize, int rank, int size) {
    if (!localChunk.isValid() || kernelSize < 3 || kernelSize % 2 == 0) {
        return Image();
    }
    
    int radius = kernelSize / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image (now has halo rows)
    Image filtered = Filters::medianFilter(extended, kernelSize);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image bilateralFilter(const Image& localChunk, int diameter, double sigmaColor, double sigmaSpace, int rank, int size) {
    if (!localChunk.isValid() || diameter < 3 || diameter % 2 == 0) {
        return Image();
    }
    
    int radius = diameter / 2;
    int haloSize = radius;  // Minimal halo size (kernel radius)
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image (now has halo rows)
    Image filtered = Filters::bilateralFilter(extended, diameter, sigmaColor, sigmaSpace);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

} // namespace FiltersMPI

