#ifndef POINT_OPERATIONS_MPI_H
#define POINT_OPERATIONS_MPI_H

#include "image.h"
#include <mpi.h>

namespace PointOpsMPI {
    // MPI versions of point operations
    // These work on local chunks - no communication needed (embarrassingly parallel)
    
    Image grayscale(const Image& localChunk, int rank, int size);
    Image adjustBrightness(const Image& localChunk, int delta, int rank, int size);
    Image adjustContrast(const Image& localChunk, float factor, int rank, int size);
    Image invert(const Image& localChunk, int rank, int size);
    Image gammaCorrection(const Image& localChunk, float gamma, int rank, int size);
    Image threshold(const Image& localChunk, uint8_t thresh, int rank, int size);
}

#endif // POINT_OPERATIONS_MPI_H

