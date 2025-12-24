#ifndef EDGE_DETECTION_MPI_H
#define EDGE_DETECTION_MPI_H

#include "image.h"
#include "edge_detection.h"

namespace EdgeDetectionMPI {
    // MPI-aware edge detection functions that use halo exchange
    
    // Sobel with MPI halo exchange (3x3 kernel, radius = 1)
    Image sobel(const Image& localChunk, int rank, int size);
    
    // Canny with MPI halo exchange
    Image canny(const Image& localChunk, double lowThreshold, double highThreshold, int rank, int size);
    
    // Sharpen with MPI halo exchange (3x3 kernel, radius = 1)
    Image sharpen(const Image& localChunk, int rank, int size);
    
    // Prewitt with MPI halo exchange (3x3 kernel, radius = 1)
    Image prewitt(const Image& localChunk, int rank, int size);
    
    // Laplacian with MPI halo exchange (3x3 kernel, radius = 1)
    Image laplacian(const Image& localChunk, int rank, int size);
}

#endif // EDGE_DETECTION_MPI_H

