#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "image.h"
#include <mpi.h>

namespace MPIUtils {
    // Distribute image across MPI processes (horizontal strip decomposition)
    // Rank 0 sends chunks to all processes, each process receives its chunk
    void distributeImage(const Image& fullImage, Image& localChunk, int rank, int size);
    
    // Gather image chunks from all processes back to rank 0
    void gatherImage(const Image& localChunk, Image& fullImage, int rank, int size);
    
    // Exchange boundary rows with neighboring processes (for filters)
    void exchangeBoundaries(Image& localChunk, int rank, int size, int boundarySize);
    
    // Get local chunk dimensions for a given rank
    void getLocalDimensions(int fullHeight, int rank, int size, 
                           int& localHeight, int& startRow);
    
    // Broadcast image metadata (width, height, channels) from rank 0
    void broadcastMetadata(int& width, int& height, int& channels, int rank);
}

#endif // MPI_UTILS_H

