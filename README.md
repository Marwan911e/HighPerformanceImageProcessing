# Image Processing Project

A comprehensive image processing library implementing various filters, transformations, and operations on images. This serial version is designed to be extended with parallel processing using OpenMP, MPI, and CUDA.

## Features

### Point Operations
- Grayscale conversion
- Brightness adjustment
- Contrast adjustment
- Thresholding / Binarization (including Otsu's method)
- Image inversion (negative)
- Gamma correction
- Adaptive thresholding (Gaussian and Mean)

### Noise Generation
- Salt-and-pepper noise
- Gaussian noise
- Speckle noise

### Smoothing / Blurring Filters
- Box blur / Average blur
- Gaussian blur
- Median filter
- Bilateral filter

### Edge Detection / Sharpening
- Sobel operator
- Canny edge detection
- Sharpen filter
- Prewitt operator
- Laplacian operator

### Morphological Operations
- Erosion
- Dilation
- Opening
- Closing
- Morphological gradient
- Top hat / Black hat

### Geometric Transformations
- Image rotation
- Scaling / Resizing (Nearest Neighbor and Bilinear)
- Translation / Shift
- Horizontal / Vertical flip
- Perspective transform

### Color / Channel Operations
- Channel splitting (R, G, B)
- Channel merging
- HSV adjustments (hue, saturation, value)
- RGB to HSV and back
- Color balance

## Building the Project

### Using CMake
```bash
mkdir build
cd build
cmake ..
make
```

### Using Makefile
```bash
make
```

## Usage

```bash
./image_processor
```

Follow the interactive menu to:
1. Load an image
2. Apply various operations
3. Save the processed image

## Image Format Support

- JPG/JPEG
- PNG
- BMP

## Project Structure

```
├── include/          # Header files
├── src/             # Source files
├── lib/             # Third-party libraries (stb_image)
├── examples/        # Sample images
├── CMakeLists.txt   # CMake build configuration
├── Makefile         # Make build configuration
└── README.md        # This file
```

## Dependencies

- C++17 compiler
- stb_image.h and stb_image_write.h (included in lib/)

## Future Extensions

This project is designed to be extended with:
- OpenMP for shared-memory parallelization
- MPI for distributed-memory parallelization
- CUDA for GPU acceleration

## License

MIT License
