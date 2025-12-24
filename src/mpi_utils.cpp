#include "mpi_utils.h"
#include <algorithm>
#include <cstring>
#include <vector>

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
    
    // Calculate local dimensions for this rank
    int localHeight, startRow;
    getLocalDimensions(height, rank, size, localHeight, startRow);
    
    // Create local chunk
    localChunk = Image(width, localHeight, channels);
    
    // Prepare arrays for MPI_Scatterv (all ranks need this)
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int rowSize = width * channels;
    
    for (int i = 0; i < size; ++i) {
        int localH, startR;
        getLocalDimensions(height, i, size, localH, startR);
        sendcounts[i] = localH * rowSize;
        displs[i] = startR * rowSize;
    }
    
    // Use MPI_Scatterv for efficient distribution
    // All processes must call MPI_Scatterv
    const uint8_t* sendbuf = (rank == 0) ? fullImage.getData() : nullptr;
    uint8_t* recvbuf = localChunk.getData();
    int recvcount = localHeight * rowSize;
    
    MPI_Scatterv(const_cast<uint8_t*>(sendbuf), sendcounts.data(), displs.data(), MPI_BYTE,
                 recvbuf, recvcount, MPI_BYTE, 0, MPI_COMM_WORLD);
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
    
    // Prepare arrays for MPI_Gatherv (all ranks need this)
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    int rowSize = width * channels;
    
    // Calculate displacements and counts for all ranks
    for (int i = 0; i < size; ++i) {
        int localH, startR;
        getLocalDimensions(height, i, size, localH, startR);
        recvcounts[i] = localH * rowSize;
        displs[i] = startR * rowSize;
    }
    
    // Use MPI_Gatherv for efficient gathering
    // All processes must call MPI_Gatherv
    const uint8_t* sendbuf = localChunk.getData();
    int sendcount = localHeight * rowSize;
    uint8_t* recvbuf = (rank == 0) ? fullImage.getData() : nullptr;
    
    MPI_Gatherv(const_cast<uint8_t*>(sendbuf), sendcount, MPI_BYTE,
                recvbuf, recvcounts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
}

// Extend image with halo rows (ghost cells) for convolution filters
Image extendWithHalo(const Image& localChunk, int rank, int size, int haloSize) {
    if (haloSize == 0) {
        return localChunk.clone();
    }
    
    int width = localChunk.getWidth();
    int height = localChunk.getHeight();
    int channels = localChunk.getChannels();
    int rowSize = width * channels;
    
    // Calculate extended dimensions
    int topHalo = (rank > 0) ? haloSize : 0;
    int bottomHalo = (rank < size - 1) ? haloSize : 0;
    int extendedHeight = height + topHalo + bottomHalo;
    
    // Create extended image
    Image extended(width, extendedHeight, channels);
    
    // Copy original data to center
    for (int y = 0; y < height; ++y) {
        const uint8_t* srcRow = localChunk.getPixel(0, y);
        uint8_t* dstRow = extended.getPixel(0, y + topHalo);
        std::memcpy(dstRow, srcRow, rowSize);
    }
    
    return extended;
}

// Exchange boundaries using non-blocking communication
// Returns extended image with halo rows
Image exchangeBoundaries(const Image& localChunk, int rank, int size, int haloSize) {
    if (haloSize == 0) {
        return localChunk.clone();
    }
    
    int width = localChunk.getWidth();
    int height = localChunk.getHeight();
    int channels = localChunk.getChannels();
    int rowSize = width * channels;
    
    // Create extended image with space for halos
    int topHalo = (rank > 0) ? haloSize : 0;
    int bottomHalo = (rank < size - 1) ? haloSize : 0;
    Image extended = extendWithHalo(localChunk, rank, size, haloSize);
    
    // Prepare non-blocking communication
    std::vector<MPI_Request> requests;
    std::vector<MPI_Status> statuses;
    
    // Exchange with upper neighbor (rank-1)
    if (rank > 0) {
        // Send top boundary rows to rank-1
        const uint8_t* topRows = localChunk.getPixel(0, 0);
        MPI_Request sendReq;
        MPI_Isend(const_cast<uint8_t*>(topRows), rowSize * haloSize, MPI_BYTE, 
                 rank - 1, 1, MPI_COMM_WORLD, &sendReq);
        requests.push_back(sendReq);
        
        // Receive boundary from rank-1 (will go into top halo)
        uint8_t* topHaloBuf = extended.getPixel(0, 0);
        MPI_Request recvReq;
        MPI_Irecv(topHaloBuf, rowSize * haloSize, MPI_BYTE, 
                 rank - 1, 2, MPI_COMM_WORLD, &recvReq);
        requests.push_back(recvReq);
    }
    
    // Exchange with lower neighbor (rank+1)
    if (rank < size - 1) {
        // Send bottom boundary rows to rank+1
        const uint8_t* bottomRows = localChunk.getPixel(0, height - haloSize);
        MPI_Request sendReq;
        MPI_Isend(const_cast<uint8_t*>(bottomRows), rowSize * haloSize, MPI_BYTE, 
                 rank + 1, 2, MPI_COMM_WORLD, &sendReq);
        requests.push_back(sendReq);
        
        // Receive boundary from rank+1 (will go into bottom halo)
        uint8_t* bottomHaloBuf = extended.getPixel(0, height + topHalo);
        MPI_Request recvReq;
        MPI_Irecv(bottomHaloBuf, rowSize * haloSize, MPI_BYTE, 
                 rank + 1, 1, MPI_COMM_WORLD, &recvReq);
        requests.push_back(recvReq);
    }
    
    // Wait for all communication to complete
    if (!requests.empty()) {
        statuses.resize(requests.size());
        MPI_Waitall(requests.size(), requests.data(), statuses.data());
    }
    
    return extended;
}

// Extract original region from extended image (remove halos)
Image removeHalo(const Image& extendedImage, int rank, int size, int haloSize) {
    if (haloSize == 0) {
        return extendedImage.clone();
    }
    
    int width = extendedImage.getWidth();
    int extendedHeight = extendedImage.getHeight();
    int channels = extendedImage.getChannels();
    
    int topHalo = (rank > 0) ? haloSize : 0;
    int bottomHalo = (rank < size - 1) ? haloSize : 0;
    int originalHeight = extendedHeight - topHalo - bottomHalo;
    
    Image result(width, originalHeight, channels);
    int rowSize = width * channels;
    
    // Copy original region (skip halos)
    for (int y = 0; y < originalHeight; ++y) {
        const uint8_t* srcRow = extendedImage.getPixel(0, y + topHalo);
        uint8_t* dstRow = result.getPixel(0, y);
        std::memcpy(dstRow, srcRow, rowSize);
    }
    
    return result;
}

} // namespace MPIUtils

