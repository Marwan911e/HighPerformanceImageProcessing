#include "mpi_utils.h"
#include <algorithm>
#include <cstring>

namespace MPIUtils {

void broadcastMetadata(int& width, int& height, int& channels, int rank) {
    int metadata[3] = {width, height, channels};
    MPI_Bcast(metadata, 3, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        width = metadata[0];
        height = metadata[1];
        channels = metadata[2];
    }
}

void getLocalDimensions(int fullHeight, int rank, int size, 
                       int& localHeight, int& startRow) {
    int rowsPerProcess = fullHeight / size;
    int remainder = fullHeight % size;
    
    // Distribute remainder rows to first 'remainder' processes
    if (rank < remainder) {
        localHeight = rowsPerProcess + 1;
        startRow = rank * localHeight;
    } else {
        localHeight = rowsPerProcess;
        startRow = remainder * (rowsPerProcess + 1) + (rank - remainder) * rowsPerProcess;
    }
}

void distributeImage(const Image& fullImage, Image& localChunk, int rank, int size) {
    int width, height, channels;
    
    if (rank == 0) {
        if (!fullImage.isValid()) {
            // Broadcast invalid signal
            width = height = channels = 0;
            broadcastMetadata(width, height, channels, rank);
            return;
        }
        width = fullImage.getWidth();
        height = fullImage.getHeight();
        channels = fullImage.getChannels();
    }
    
    // Broadcast metadata
    broadcastMetadata(width, height, channels, rank);
    
    if (width == 0 || height == 0) {
        // Invalid image
        return;
    }
    
    // Calculate local dimensions
    int localHeight, startRow;
    getLocalDimensions(height, rank, size, localHeight, startRow);
    
    // Create local chunk
    localChunk = Image(width, localHeight, channels);
    
    if (rank == 0) {
        // Rank 0 sends chunks to all processes
        for (int dest = 0; dest < size; ++dest) {
            int destHeight, destStartRow;
            getLocalDimensions(height, dest, size, destHeight, destStartRow);
            
            if (dest == 0) {
                // Copy directly for rank 0
                for (int y = 0; y < destHeight; ++y) {
                    const uint8_t* srcRow = fullImage.getPixel(0, destStartRow + y);
                    uint8_t* dstRow = localChunk.getPixel(0, y);
                    std::memcpy(dstRow, srcRow, width * channels);
                }
            } else {
                // Send to other processes
                for (int y = 0; y < destHeight; ++y) {
                    const uint8_t* row = fullImage.getPixel(0, destStartRow + y);
                    MPI_Send(const_cast<uint8_t*>(row), width * channels, MPI_BYTE, 
                            dest, 0, MPI_COMM_WORLD);
                }
            }
        }
    } else {
        // Other ranks receive their chunk
        for (int y = 0; y < localHeight; ++y) {
            uint8_t* row = localChunk.getPixel(0, y);
            MPI_Recv(row, width * channels, MPI_BYTE, 0, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void gatherImage(const Image& localChunk, Image& fullImage, int rank, int size) {
    int width, height, channels;
    int localHeight = localChunk.getHeight();
    
    if (rank == 0) {
        width = localChunk.getWidth();
        channels = localChunk.getChannels();
    }
    
    // Broadcast metadata
    broadcastMetadata(width, height, channels, rank);
    
    // Calculate total height by summing all local heights
    int totalHeight = 0;
    MPI_Allreduce(&localHeight, &totalHeight, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == 0) {
        height = totalHeight;
        fullImage = Image(width, height, channels);
    }
    
    // Gather chunks
    if (rank == 0) {
        // Rank 0 receives from all processes
        for (int src = 0; src < size; ++src) {
            int srcHeight, srcStartRow;
            getLocalDimensions(height, src, size, srcHeight, srcStartRow);
            
            if (src == 0) {
                // Copy directly from rank 0
                for (int y = 0; y < srcHeight; ++y) {
                    const uint8_t* srcRow = localChunk.getPixel(0, y);
                    uint8_t* dstRow = fullImage.getPixel(0, srcStartRow + y);
                    std::memcpy(dstRow, srcRow, width * channels);
                }
            } else {
                // Receive from other processes
                for (int y = 0; y < srcHeight; ++y) {
                    uint8_t* row = fullImage.getPixel(0, srcStartRow + y);
                    MPI_Recv(row, width * channels, MPI_BYTE, src, 0, 
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    } else {
        // Other ranks send their chunk
        for (int y = 0; y < localHeight; ++y) {
            const uint8_t* row = localChunk.getPixel(0, y);
            MPI_Send(const_cast<uint8_t*>(row), width * channels, MPI_BYTE, 
                    0, 0, MPI_COMM_WORLD);
        }
    }
}

void exchangeBoundaries(Image& localChunk, int rank, int size, int boundarySize) {
    int width = localChunk.getWidth();
    int height = localChunk.getHeight();
    int channels = localChunk.getChannels();
    int rowSize = width * channels;
    
    // Exchange with upper neighbor (send top, receive from above)
    if (rank > 0) {
        // Send top boundary to rank-1
        uint8_t* topRow = localChunk.getPixel(0, 0);
        MPI_Send(topRow, rowSize * boundarySize, MPI_BYTE, rank - 1, 1, MPI_COMM_WORLD);
        
        // Receive boundary from rank-1 (will be prepended)
        // Note: This requires extending the image, simplified version here
    }
    
    // Exchange with lower neighbor (send bottom, receive from below)
    if (rank < size - 1) {
        // Send bottom boundary to rank+1
        uint8_t* bottomRow = localChunk.getPixel(0, height - boundarySize);
        MPI_Send(bottomRow, rowSize * boundarySize, MPI_BYTE, rank + 1, 2, MPI_COMM_WORLD);
        
        // Receive boundary from rank+1 (will be appended)
        // Note: This requires extending the image, simplified version here
    }
    
    // For now, this is a placeholder - full implementation would require
    // creating extended image with ghost cells
}

} // namespace MPIUtils

