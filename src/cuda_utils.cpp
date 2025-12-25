#include "cuda_utils.h"
#include <iostream>
#include <algorithm>

namespace CudaUtils {

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line 
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

// StreamManager implementation
StreamManager::StreamManager(int numStreams) : numStreams(numStreams) {
    streams.resize(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
}

StreamManager::~StreamManager() {
    for (auto& stream : streams) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
}

cudaStream_t StreamManager::getStream(int index) const {
    return streams[index % numStreams];
}

void StreamManager::synchronizeAll() const {
    for (const auto& stream : streams) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

// PinnedBuffer implementation
PinnedBuffer::PinnedBuffer() : hostPtr(nullptr), bufferSize(0) {}

PinnedBuffer::~PinnedBuffer() {
    deallocate();
}

bool PinnedBuffer::allocate(size_t size) {
    if (hostPtr && bufferSize >= size) {
        return true; // Already allocated with sufficient size
    }
    
    deallocate();
    CUDA_CHECK(cudaHostAlloc((void**)&hostPtr, size, cudaHostAllocDefault));
    bufferSize = size;
    return true;
}

void PinnedBuffer::deallocate() {
    if (hostPtr) {
        cudaFreeHost(hostPtr);
        hostPtr = nullptr;
        bufferSize = 0;
    }
}

// DeviceBuffer implementation
DeviceBuffer::DeviceBuffer() : devicePtr(nullptr), bufferSize(0) {}

DeviceBuffer::~DeviceBuffer() {
    deallocate();
}

bool DeviceBuffer::allocate(size_t size) {
    if (devicePtr && bufferSize >= size) {
        return true; // Already allocated with sufficient size
    }
    
    deallocate();
    CUDA_CHECK(cudaMalloc((void**)&devicePtr, size));
    bufferSize = size;
    return true;
}

void DeviceBuffer::deallocate() {
    if (devicePtr) {
        cudaFree(devicePtr);
        devicePtr = nullptr;
        bufferSize = 0;
    }
}

// Generate tiles for overlap processing
std::vector<Tile> generateTiles(int imageWidth, int imageHeight,
                                int tileWidth, int tileHeight,
                                int overlap) {
    std::vector<Tile> tiles;
    
    int stepX = tileWidth - overlap;
    int stepY = tileHeight - overlap;
    
    int tileIndex = 0;
    for (int y = 0; y < imageHeight; y += stepY) {
        for (int x = 0; x < imageWidth; x += stepX) {
            Tile tile;
            tile.startX = x;
            tile.startY = y;
            tile.width = std::min(tileWidth, imageWidth - x);
            tile.height = std::min(tileHeight, imageHeight - y);
            tile.tileIndex = tileIndex++;
            tiles.push_back(tile);
        }
    }
    
    return tiles;
}

// Copy tile to device (async)
void copyTileToDevice(const Image& src, const Tile& tile,
                     uint8_t* d_dst, int imageWidth, int imageChannels,
                     cudaStream_t stream) {
    size_t tilePitch = tile.width * imageChannels;
    size_t imagePitch = imageWidth * imageChannels;
    
    // Copy row by row from host to device
    for (int y = 0; y < tile.height; ++y) {
        const uint8_t* srcRow = src.getPixel(tile.startX, tile.startY + y);
        uint8_t* dstRow = d_dst + ((tile.startY + y) * imageWidth + tile.startX) * imageChannels;
        
        CUDA_CHECK(cudaMemcpyAsync(dstRow, srcRow, tilePitch,
                                   cudaMemcpyHostToDevice, stream));
    }
}

// Copy tile from device (async)
void copyTileFromDevice(uint8_t* d_src, Image& dst, const Tile& tile,
                       int imageWidth, int imageChannels,
                       cudaStream_t stream) {
    size_t tilePitch = tile.width * imageChannels;
    
    // Copy row by row from device to host
    for (int y = 0; y < tile.height; ++y) {
        uint8_t* dstRow = dst.getPixel(tile.startX, tile.startY + y);
        const uint8_t* srcRow = d_src + ((tile.startY + y) * imageWidth + tile.startX) * imageChannels;
        
        CUDA_CHECK(cudaMemcpyAsync(dstRow, srcRow, tilePitch,
                                   cudaMemcpyDeviceToHost, stream));
    }
}

// Copy full image to device (async, pinned memory)
void copyImageToDevice(const Image& src, uint8_t* d_dst, cudaStream_t stream) {
    size_t size = src.getDataSize();
    CUDA_CHECK(cudaMemcpyAsync(d_dst, src.getData(), size,
                               cudaMemcpyHostToDevice, stream));
}

// Copy full image from device (async, pinned memory)
void copyImageFromDevice(uint8_t* d_src, Image& dst, cudaStream_t stream) {
    size_t size = dst.getDataSize();
    CUDA_CHECK(cudaMemcpyAsync(dst.getData(), d_src, size,
                               cudaMemcpyDeviceToHost, stream));
}

} // namespace CudaUtils

