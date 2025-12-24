#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "image.h"
#include <mpi.h>

namespace MPIUtils {
    // Distribute image across MPI processes (horizontal strip decomposition)
    // Uses MPI_Scatterv for efficient distribution
    void distributeImage(const Image& fullImage, Image& localChunk, int rank, int size);
    
    // Gather image chunks from all processes back to rank 0
    // Uses MPI_Gatherv for efficient gathering
    void gatherImage(const Image& localChunk, Image& fullImage, int rank, int size);
    
    // Exchange boundary rows with neighboring processes using non-blocking communication
    // Returns extended image with halo/ghost rows for convolution filters
    // Uses MPI_Isend/MPI_Irecv/MPI_Waitall for overlap of communication and computation
    Image exchangeBoundaries(const Image& localChunk, int rank, int size, int haloSize);
    
    // Extend image with halo rows (ghost cells) for convolution filters
    Image extendWithHalo(const Image& localChunk, int rank, int size, int haloSize);
    
    // Remove halo rows from extended image to get original region
    Image removeHalo(const Image& extendedImage, int rank, int size, int haloSize);
    
    // Get local chunk dimensions for a given rank
    void getLocalDimensions(int fullHeight, int rank, int size, 
                           int& localHeight, int& startRow);
    
    // Broadcast image metadata (width, height, channels) from rank 0
    void broadcastMetadata(int& width, int& height, int& channels, int rank);
}

#endif // MPI_UTILS_H

