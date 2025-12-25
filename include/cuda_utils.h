#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include "image.h"
#include <cuda_runtime.h>
#include <vector>

namespace CudaUtils {
    // Error checking
    void checkCudaError(cudaError_t err, const char* file, int line);
    #define CUDA_CHECK(err) CudaUtils::checkCudaError(err, __FILE__, __LINE__)

    // Stream management
    class StreamManager {
    public:
        StreamManager(int numStreams = 4);
        ~StreamManager();
        
        cudaStream_t getStream(int index) const;
        int getNumStreams() const { return numStreams; }
        void synchronizeAll() const;
        
    private:
        int numStreams;
        std::vector<cudaStream_t> streams;
    };

    // Pinned memory buffer
    class PinnedBuffer {
    public:
        PinnedBuffer();
        ~PinnedBuffer();
        
        bool allocate(size_t size);
        void deallocate();
        
        uint8_t* getHostPtr() { return hostPtr; }
        const uint8_t* getHostPtr() const { return hostPtr; }
        size_t getSize() const { return bufferSize; }
        
    private:
        uint8_t* hostPtr;
        size_t bufferSize;
    };

    // Device memory buffer
    class DeviceBuffer {
    public:
        DeviceBuffer();
        ~DeviceBuffer();
        
        bool allocate(size_t size);
        void deallocate();
        
        uint8_t* getDevicePtr() { return devicePtr; }
        const uint8_t* getDevicePtr() const { return devicePtr; }
        size_t getSize() const { return bufferSize; }
        
    private:
        uint8_t* devicePtr;
        size_t bufferSize;
    };

    // Tile structure for overlap strategy
    struct Tile {
        int startX, startY;
        int width, height;
        int tileIndex;
    };

    // Generate tiles for overlap processing
    std::vector<Tile> generateTiles(int imageWidth, int imageHeight, 
                                    int tileWidth, int tileHeight, 
                                    int overlap);

    // Copy image tile to device (async)
    void copyTileToDevice(const Image& src, const Tile& tile, 
                         uint8_t* d_dst, int imageWidth, int imageChannels,
                         cudaStream_t stream);

    // Copy image tile from device (async)
    void copyTileFromDevice(uint8_t* d_src, Image& dst, const Tile& tile,
                           int imageWidth, int imageChannels,
                           cudaStream_t stream);

    // Copy full image to device (async, pinned memory)
    void copyImageToDevice(const Image& src, uint8_t* d_dst, 
                          cudaStream_t stream = 0);

    // Copy full image from device (async, pinned memory)
    void copyImageFromDevice(uint8_t* d_src, Image& dst,
                            cudaStream_t stream = 0);
}

#endif // CUDA_UTILS_H

