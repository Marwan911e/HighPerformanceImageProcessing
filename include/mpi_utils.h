#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include "image.h"
#include <mpi.h>
#include <vector>

namespace MPIUtils {
    // Structure to hold non-blocking communication state for overlap
    struct BoundaryExchange {
        Image extended;
        std::vector<MPI_Request> requests;
        int rank;
        int size;
        int haloSize;
    };
    
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
    
    // ===== NEW: Functions for computation-communication overlap =====
    
    // Start non-blocking boundary exchange (returns immediately)
    // Returns BoundaryExchange struct containing extended image and MPI requests
    // Use this to overlap computation with communication
    BoundaryExchange startBoundaryExchange(const Image& localChunk, int rank, int size, int haloSize);
    
    // Wait for boundary exchange to complete
    // Call this after computing inner region, before computing border pixels
    void waitBoundaryExchange(BoundaryExchange& exchange);
    
    // Get inner region boundaries (region that doesn't need halo data)
    // Returns {startY, endY} in the extended image coordinates
    void getInnerRegion(int localHeight, int haloSize, int rank, int size, 
                       int& startY, int& endY);
    
    // Get border region boundaries (region that needs halo data)
    // Returns two ranges: top border and bottom border
    // topStart, topEnd, bottomStart, bottomEnd in extended image coordinates
    void getBorderRegions(int localHeight, int haloSize, int rank, int size,
                         int& topStart, int& topEnd, int& bottomStart, int& bottomEnd);
    
    // ===== End new functions =====
    
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

