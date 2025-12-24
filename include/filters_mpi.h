#ifndef FILTERS_MPI_H
#define FILTERS_MPI_H

#include "image.h"
#include "filters.h"

namespace FiltersMPI {
    // MPI-aware filter functions that use halo exchange
    // These functions handle boundary exchange automatically
    
    // Box blur with MPI halo exchange
    Image boxBlur(const Image& localChunk, int kernelSize, int rank, int size);
    
    // Gaussian blur with MPI halo exchange
    Image gaussianBlur(const Image& localChunk, int kernelSize, float sigma, int rank, int size);
    
    // Median filter with MPI halo exchange
    Image medianFilter(const Image& localChunk, int kernelSize, int rank, int size);
    
    // Bilateral filter with MPI halo exchange
    Image bilateralFilter(const Image& localChunk, int diameter, double sigmaColor, double sigmaSpace, int rank, int size);
}

#endif // FILTERS_MPI_H

