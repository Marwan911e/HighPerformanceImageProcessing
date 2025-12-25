#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for kernel launchers
void launchBoxBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                        int width, int height, int channels, int kernelSize,
                        cudaStream_t stream);

void launchGaussianBlurKernel(const uint8_t* d_input, uint8_t* d_output,
                             int width, int height, int channels,
                             const float* kernel, int kernelSize,
                             cudaStream_t stream);

void launchSobelKernel(const uint8_t* d_input, uint8_t* d_output,
                      int width, int height, int channels,
                      cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_CUH

