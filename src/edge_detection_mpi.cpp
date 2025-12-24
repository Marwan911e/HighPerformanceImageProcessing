#include "edge_detection_mpi.h"
#include "mpi_utils.h"
#include "edge_detection.h"

namespace EdgeDetectionMPI {

Image sobel(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image
    Image filtered = EdgeDetection::sobel(extended);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image canny(const Image& localChunk, double lowThreshold, double highThreshold, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // Canny uses Sobel (3x3 kernel, radius = 1)
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image
    Image filtered = EdgeDetection::canny(extended, lowThreshold, highThreshold);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image sharpen(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image
    Image filtered = EdgeDetection::sharpen(extended);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image prewitt(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image
    Image filtered = EdgeDetection::prewitt(extended);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

Image laplacian(const Image& localChunk, int rank, int size) {
    if (!localChunk.isValid()) {
        return Image();
    }
    
    int haloSize = 1;  // 3x3 kernel, radius = 1
    
    // Exchange boundaries with non-blocking communication
    Image extended = MPIUtils::exchangeBoundaries(localChunk, rank, size, haloSize);
    
    // Apply filter on extended image
    Image filtered = EdgeDetection::laplacian(extended);
    
    // Remove halo rows to get original region
    return MPIUtils::removeHalo(filtered, rank, size, haloSize);
}

} // namespace EdgeDetectionMPI

