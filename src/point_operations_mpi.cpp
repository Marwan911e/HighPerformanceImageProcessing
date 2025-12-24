#include "point_operations_mpi.h"
#include "point_operations.h"
#include <algorithm>
#include <cmath>

namespace PointOpsMPI {

// MPI version of grayscale - works on local chunk
Image grayscale(const Image& localChunk, int rank, int size) {
    // This is embarrassingly parallel - just process local chunk
    return PointOps::grayscale(localChunk);
}

// MPI version of brightness adjustment
Image adjustBrightness(const Image& localChunk, int delta, int rank, int size) {
    return PointOps::adjustBrightness(localChunk, delta);
}

// MPI version of contrast adjustment
Image adjustContrast(const Image& localChunk, float factor, int rank, int size) {
    return PointOps::adjustContrast(localChunk, factor);
}

// MPI version of invert
Image invert(const Image& localChunk, int rank, int size) {
    return PointOps::invert(localChunk);
}

// MPI version of gamma correction
Image gammaCorrection(const Image& localChunk, float gamma, int rank, int size) {
    return PointOps::gammaCorrection(localChunk, gamma);
}

// MPI version of threshold
Image threshold(const Image& localChunk, uint8_t thresh, int rank, int size) {
    return PointOps::threshold(localChunk, thresh);
}

} // namespace PointOpsMPI

