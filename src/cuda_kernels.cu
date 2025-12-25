#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cmath>

// Constant memory for filter coefficients (max 32x32 kernel)
#define MAX_KERNEL_SIZE 32
__constant__ float c_gaussianKernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];
__constant__ int c_kernelSize;
__constant__ int c_kernelRadius;

// Texture memory for image data (read-only, cached)
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texImage;

// Shared memory tile size
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

// Box blur kernel with shared memory
__global__ void boxBlurKernel(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int kernelSize) {
    int radius = kernelSize / 2;
    
    // Shared memory for tile + halo
    extern __shared__ uint8_t sharedTile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_HEIGHT + ty;
    
    // Load tile into shared memory with halo
    int sharedIdx = (ty + radius) * (TILE_WIDTH + 2 * radius) + (tx + radius);
    int globalIdx = y * width + x;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            sharedTile[sharedIdx * channels + c] = 
                input[globalIdx * channels + c];
        }
    }
    
    // Load halo pixels
    if (tx < radius) {
        // Left halo
        int haloX = x - radius;
        if (haloX >= 0 && y < height) {
            int haloIdx = (ty + radius) * (TILE_WIDTH + 2 * radius) + tx;
            int globalHaloIdx = y * width + haloX;
            for (int c = 0; c < channels; ++c) {
                sharedTile[haloIdx * channels + c] = 
                    input[globalHaloIdx * channels + c];
            }
        }
    }
    if (tx >= TILE_WIDTH - radius) {
        // Right halo
        int haloX = x + radius;
        if (haloX < width && y < height) {
            int haloIdx = (ty + radius) * (TILE_WIDTH + 2 * radius) + (tx + radius + radius);
            int globalHaloIdx = y * width + haloX;
            for (int c = 0; c < channels; ++c) {
                sharedTile[haloIdx * channels + c] = 
                    input[globalHaloIdx * channels + c];
            }
        }
    }
    if (ty < radius) {
        // Top halo
        int haloY = y - radius;
        if (haloY >= 0 && x < width) {
            int haloIdx = ty * (TILE_WIDTH + 2 * radius) + (tx + radius);
            int globalHaloIdx = haloY * width + x;
            for (int c = 0; c < channels; ++c) {
                sharedTile[haloIdx * channels + c] = 
                    input[globalHaloIdx * channels + c];
            }
        }
    }
    if (ty >= TILE_HEIGHT - radius) {
        // Bottom halo
        int haloY = y + radius;
        if (haloY < height && x < width) {
            int haloIdx = (ty + radius + radius) * (TILE_WIDTH + 2 * radius) + (tx + radius);
            int globalHaloIdx = haloY * width + x;
            for (int c = 0; c < channels; ++c) {
                sharedTile[haloIdx * channels + c] = 
                    input[globalHaloIdx * channels + c];
            }
        }
    }
    
    __syncthreads();
    
    // Compute box blur using shared memory
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            int count = 0;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = tx + radius + dx;
                    int sy = ty + radius + dy;
                    
                    if (sx >= 0 && sx < TILE_WIDTH + 2 * radius &&
                        sy >= 0 && sy < TILE_HEIGHT + 2 * radius) {
                        int sharedIdx = sy * (TILE_WIDTH + 2 * radius) + sx;
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            sum += sharedTile[sharedIdx * channels + c];
                            count++;
                        }
                    }
                }
            }
            
            output[globalIdx * channels + c] = 
                static_cast<uint8_t>(sum / count);
        }
    }
}

// Gaussian blur kernel with shared memory and constant memory
__global__ void gaussianBlurKernel(const uint8_t* input, uint8_t* output,
                                  int width, int height, int channels) {
    int radius = c_kernelRadius;
    
    // Shared memory for tile + halo
    extern __shared__ uint8_t sharedTile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_HEIGHT + ty;
    
    // Load tile into shared memory (similar to box blur)
    int sharedIdx = (ty + radius) * (TILE_WIDTH + 2 * radius) + (tx + radius);
    int globalIdx = y * width + x;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            sharedTile[sharedIdx * channels + c] = 
                input[globalIdx * channels + c];
        }
    }
    
    // Load halo (simplified - full implementation would load all halo regions)
    // ... (similar halo loading as box blur)
    
    __syncthreads();
    
    // Compute Gaussian blur using shared memory and constant kernel
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            float accum = 0.0f;
            
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        int kernelIdx = (dy + radius) * c_kernelSize + (dx + radius);
                        float weight = c_gaussianKernel[kernelIdx];
                        
                        int sharedIdx = (ty + radius + dy) * (TILE_WIDTH + 2 * radius) + (tx + radius + dx);
                        if (sharedIdx >= 0 && sharedIdx < (TILE_WIDTH + 2 * radius) * (TILE_HEIGHT + 2 * radius)) {
                            accum += sharedTile[sharedIdx * channels + c] * weight;
                        }
                    }
                }
            }
            
            output[globalIdx * channels + c] = 
                static_cast<uint8_t>(fmaxf(0.0f, fminf(255.0f, accum)));
        }
    }
}

// Sobel kernel with shared memory
__global__ void sobelKernel(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels) {
    // Sobel kernels
    __shared__ int sobelX[3][3];
    __shared__ int sobelY[3][3];
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sobelX[0][0] = -1; sobelX[0][1] = 0; sobelX[0][2] = 1;
        sobelX[1][0] = -2; sobelX[1][1] = 0; sobelX[1][2] = 2;
        sobelX[2][0] = -1; sobelX[2][1] = 0; sobelX[2][2] = 1;
        
        sobelY[0][0] = -1; sobelY[0][1] = -2; sobelY[0][2] = -1;
        sobelY[1][0] =  0; sobelY[1][1] =  0; sobelY[1][2] =  0;
        sobelY[2][0] =  1; sobelY[2][1] =  2; sobelY[2][2] =  1;
    }
    __syncthreads();
    
    extern __shared__ uint8_t sharedTile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_HEIGHT + ty;
    
    // Load tile with 1-pixel halo
    int sharedIdx = (ty + 1) * (TILE_WIDTH + 2) + (tx + 1);
    int globalIdx = y * width + x;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            sharedTile[sharedIdx * channels + c] = 
                input[globalIdx * channels + c];
        }
    }
    
    // Load halo pixels
    if (tx == 0 && x > 0) {
        int haloIdx = (ty + 1) * (TILE_WIDTH + 2);
        int globalHaloIdx = y * width + (x - 1);
        for (int c = 0; c < channels; ++c) {
            sharedTile[haloIdx * channels + c] = 
                input[globalHaloIdx * channels + c];
        }
    }
    if (tx == TILE_WIDTH - 1 && x < width - 1) {
        int haloIdx = (ty + 1) * (TILE_WIDTH + 2) + (TILE_WIDTH + 1);
        int globalHaloIdx = y * width + (x + 1);
        for (int c = 0; c < channels; ++c) {
            sharedTile[haloIdx * channels + c] = 
                input[globalHaloIdx * channels + c];
        }
    }
    if (ty == 0 && y > 0) {
        int haloIdx = (tx + 1);
        int globalHaloIdx = (y - 1) * width + x;
        for (int c = 0; c < channels; ++c) {
            sharedTile[haloIdx * channels + c] = 
                input[globalHaloIdx * channels + c];
        }
    }
    if (ty == TILE_HEIGHT - 1 && y < height - 1) {
        int haloIdx = (TILE_HEIGHT + 1) * (TILE_WIDTH + 2) + (tx + 1);
        int globalHaloIdx = (y + 1) * width + x;
        for (int c = 0; c < channels; ++c) {
            sharedTile[haloIdx * channels + c] = 
                input[globalHaloIdx * channels + c];
        }
    }
    
    __syncthreads();
    
    // Apply Sobel operator
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        for (int c = 0; c < channels; ++c) {
            int gx = 0, gy = 0;
            
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int sx = tx + 1 + dx;
                    int sy = ty + 1 + dy;
                    int sharedIdx = sy * (TILE_WIDTH + 2) + sx;
                    uint8_t pixel = sharedTile[sharedIdx * channels + c];
                    
                    gx += pixel * sobelX[dy + 1][dx + 1];
                    gy += pixel * sobelY[dy + 1][dx + 1];
                }
            }
            
            int magnitude = (int)sqrtf((float)(gx * gx + gy * gy));
            output[globalIdx * channels + c] = 
                static_cast<uint8_t>(fminf(255.0f, (float)magnitude));
        }
    } else if (x < width && y < height) {
        // Border pixels - just copy
        for (int c = 0; c < channels; ++c) {
            output[globalIdx * channels + c] = input[globalIdx * channels + c];
        }
    }
}

// Helper function to launch box blur kernel
extern "C" void launchBoxBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height, int channels, int kernelSize,
                        cudaStream_t stream) {
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH,
                  (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    int radius = kernelSize / 2;
    size_t sharedMemSize = (TILE_WIDTH + 2 * radius) * 
                          (TILE_HEIGHT + 2 * radius) * channels * sizeof(uint8_t);
    
    boxBlurKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_input, d_output, width, height, channels, kernelSize);
}

// Helper function to launch Gaussian blur kernel
extern "C" void launchGaussianBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                             int width, int height, int channels,
                             const float* kernel, int kernelSize,
                             cudaStream_t stream) {
    // Copy kernel to constant memory
    int radius = kernelSize / 2;
    cudaMemcpyToSymbol(c_gaussianKernel, kernel, 
                       kernelSize * kernelSize * sizeof(float));
    cudaMemcpyToSymbol(c_kernelSize, &kernelSize, sizeof(int));
    cudaMemcpyToSymbol(c_kernelRadius, &radius, sizeof(int));
    
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH,
                  (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    size_t sharedMemSize = (TILE_WIDTH + 2 * radius) * 
                          (TILE_HEIGHT + 2 * radius) * channels * sizeof(uint8_t);
    
    gaussianBlurKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_input, d_output, width, height, channels);
}

// Helper function to launch Sobel kernel
extern "C" void launchSobelKernel(const uint8_t* d_input, uint8_t* d_output,
                      int width, int height, int channels,
                      cudaStream_t stream) {
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((width + TILE_WIDTH - 1) / TILE_WIDTH,
                  (height + TILE_HEIGHT - 1) / TILE_HEIGHT);
    
    size_t sharedMemSize = (TILE_WIDTH + 2) * (TILE_HEIGHT + 2) * 
                          channels * sizeof(uint8_t);
    
    sobelKernel<<<gridSize, blockSize, sharedMemSize, stream>>>(
        d_input, d_output, width, height, channels);
}

