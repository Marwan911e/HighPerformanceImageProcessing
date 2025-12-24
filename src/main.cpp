#include "image.h"
#include "point_operations.h"
#include "noise.h"
#include "filters.h"
#include "edge_detection.h"
#include "morphological.h"
#include "geometric.h"
#include "color_operations.h"
#include <iostream>
#include <string>
#include <iomanip>

void displayMenu() {
    std::cout << "\n=== IMAGE PROCESSING OPERATIONS ===\n";
    std::cout << "1.  Load Image\n";
    std::cout << "2.  Save Image\n";
    std::cout << "\nPoint Operations:\n";
    std::cout << "10. Grayscale Conversion\n";
    std::cout << "11. Adjust Brightness\n";
    std::cout << "12. Adjust Contrast\n";
    std::cout << "13. Threshold (Manual)\n";
    std::cout << "14. Threshold (Otsu)\n";
    std::cout << "15. Adaptive Threshold\n";
    std::cout << "16. Invert\n";
    std::cout << "17. Gamma Correction\n";
    std::cout << "\nNoise:\n";
    std::cout << "20. Add Salt & Pepper Noise\n";
    std::cout << "21. Add Gaussian Noise\n";
    std::cout << "22. Add Speckle Noise\n";
    std::cout << "\nFilters:\n";
    std::cout << "30. Box Blur\n";
    std::cout << "31. Gaussian Blur\n";
    std::cout << "32. Median Filter\n";
    std::cout << "33. Bilateral Filter\n";
    std::cout << "\nEdge Detection:\n";
    std::cout << "40. Sobel\n";
    std::cout << "41. Canny\n";
    std::cout << "42. Sharpen\n";
    std::cout << "43. Prewitt\n";
    std::cout << "44. Laplacian\n";
    std::cout << "\nMorphological:\n";
    std::cout << "50. Erosion\n";
    std::cout << "51. Dilation\n";
    std::cout << "52. Opening\n";
    std::cout << "53. Closing\n";
    std::cout << "54. Morphological Gradient\n";
    std::cout << "\nGeometric:\n";
    std::cout << "60. Rotate\n";
    std::cout << "61. Scale/Resize\n";
    std::cout << "62. Translate\n";
    std::cout << "63. Flip Horizontal\n";
    std::cout << "64. Flip Vertical\n";
    std::cout << "\nColor Operations:\n";
    std::cout << "70. Split Channels\n";
    std::cout << "71. RGB to HSV\n";
    std::cout << "72. Adjust Hue\n";
    std::cout << "73. Adjust Saturation\n";
    std::cout << "74. Adjust Value\n";
    std::cout << "75. Color Balance\n";
    std::cout << "\n0.  Exit\n";
    std::cout << "===================================\n";
    std::cout << "Enter choice: ";
}

int main() {
    Image currentImage;
    std::string currentFilename;
    
    std::cout << "========================================\n";
    std::cout << "  IMAGE PROCESSING APPLICATION (OpenMP)\n";
    std::cout << "========================================\n";
    std::cout << "\nSupported formats: JPG, PNG, BMP, TGA\n";
    
    while (true) {
        displayMenu();
        
        int choice;
        std::cin >> choice;
        
        if (choice == 0) {
            std::cout << "\nExiting...\n";
            break;
        }
        
        switch (choice) {
            case 1: {
                // Load Image
                std::cout << "Enter image filename: ";
                std::cin >> currentFilename;
                if (currentImage.load(currentFilename)) {
                    std::cout << "Image loaded successfully!\n";
                    std::cout << "Dimensions: " << currentImage.getWidth() << "x" << currentImage.getHeight();
                    std::cout << ", Channels: " << currentImage.getChannels() << "\n";
                } else {
                    std::cout << "Error: Failed to load image.\n";
                }
                break;
            }
            
            case 2: {
                // Save Image
                if (!currentImage.isValid()) {
                    std::cout << "Error: No image loaded.\n";
                    break;
                }
                std::string filename;
                std::cout << "Enter output filename (with extension): ";
                std::cin >> filename;
                if (currentImage.save(filename)) {
                    std::cout << "Image saved successfully!\n";
                } else {
                    std::cout << "Error: Failed to save image.\n";
                }
                break;
            }
            
            case 10: {
                // Grayscale
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = PointOps::grayscale(currentImage);
                std::cout << "Converted to grayscale.\n";
                break;
            }
            
            case 11: {
                // Brightness
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int delta;
                std::cout << "Enter brightness delta (-255 to 255): ";
                std::cin >> delta;
                currentImage = PointOps::adjustBrightness(currentImage, delta);
                std::cout << "Brightness adjusted.\n";
                break;
            }
            
            case 12: {
                // Contrast
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float factor;
                std::cout << "Enter contrast factor (0.0 to 3.0): ";
                std::cin >> factor;
                currentImage = PointOps::adjustContrast(currentImage, factor);
                std::cout << "Contrast adjusted.\n";
                break;
            }
            
            case 13: {
                // Manual Threshold
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int thresh;
                std::cout << "Enter threshold value (0-255): ";
                std::cin >> thresh;
                currentImage = PointOps::threshold(currentImage, thresh);
                std::cout << "Threshold applied.\n";
                break;
            }
            
            case 14: {
                // Otsu Threshold
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = PointOps::thresholdOtsu(currentImage);
                std::cout << "Otsu threshold applied.\n";
                break;
            }
            
            case 15: {
                // Adaptive Threshold
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int blockSize;
                std::cout << "Enter block size (odd number, e.g., 11): ";
                std::cin >> blockSize;
                currentImage = PointOps::adaptiveThreshold(currentImage, blockSize, 2, true);
                std::cout << "Adaptive threshold applied.\n";
                break;
            }
            
            case 16: {
                // Invert
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = PointOps::invert(currentImage);
                std::cout << "Image inverted.\n";
                break;
            }
            
            case 17: {
                // Gamma Correction
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float gamma;
                std::cout << "Enter gamma value (e.g., 0.5, 1.0, 2.2): ";
                std::cin >> gamma;
                currentImage = PointOps::gammaCorrection(currentImage, gamma);
                std::cout << "Gamma correction applied.\n";
                break;
            }
            
            case 20: {
                // Salt & Pepper
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float amount;
                std::cout << "Enter noise amount (0.0 to 1.0): ";
                std::cin >> amount;
                currentImage = Noise::saltAndPepper(currentImage, amount);
                std::cout << "Salt & pepper noise added.\n";
                break;
            }
            
            case 21: {
                // Gaussian Noise
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float mean, stddev;
                std::cout << "Enter mean (e.g., 0): ";
                std::cin >> mean;
                std::cout << "Enter standard deviation (e.g., 25): ";
                std::cin >> stddev;
                currentImage = Noise::gaussian(currentImage, mean, stddev);
                std::cout << "Gaussian noise added.\n";
                break;
            }
            
            case 22: {
                // Speckle Noise
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float variance;
                std::cout << "Enter variance (e.g., 0.1): ";
                std::cin >> variance;
                currentImage = Noise::speckle(currentImage, variance);
                std::cout << "Speckle noise added.\n";
                break;
            }
            
            case 30: {
                // Box Blur
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int size;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                currentImage = Filters::boxBlur(currentImage, size);
                std::cout << "Box blur applied.\n";
                break;
            }
            
            case 31: {
                // Gaussian Blur
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int size;
                float sigma;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                std::cout << "Enter sigma (e.g., 1.4): ";
                std::cin >> sigma;
                currentImage = Filters::gaussianBlur(currentImage, size, sigma);
                std::cout << "Gaussian blur applied.\n";
                break;
            }
            
            case 32: {
                // Median Filter
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int size;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                currentImage = Filters::medianFilter(currentImage, size);
                std::cout << "Median filter applied.\n";
                break;
            }
            
            case 33: {
                // Bilateral Filter
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int diameter;
                double sigmaColor, sigmaSpace;
                std::cout << "Enter diameter (e.g., 9): ";
                std::cin >> diameter;
                std::cout << "Enter sigma color (e.g., 75): ";
                std::cin >> sigmaColor;
                std::cout << "Enter sigma space (e.g., 75): ";
                std::cin >> sigmaSpace;
                currentImage = Filters::bilateralFilter(currentImage, diameter, sigmaColor, sigmaSpace);
                std::cout << "Bilateral filter applied.\n";
                break;
            }
            
            case 40: {
                // Sobel
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = EdgeDetection::sobel(currentImage);
                std::cout << "Sobel edge detection applied.\n";
                break;
            }
            
            case 41: {
                // Canny
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                double low, high;
                std::cout << "Enter low threshold (e.g., 50): ";
                std::cin >> low;
                std::cout << "Enter high threshold (e.g., 150): ";
                std::cin >> high;
                currentImage = EdgeDetection::canny(currentImage, low, high);
                std::cout << "Canny edge detection applied.\n";
                break;
            }
            
            case 42: {
                // Sharpen
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = EdgeDetection::sharpen(currentImage);
                std::cout << "Sharpen filter applied.\n";
                break;
            }
            
            case 43: {
                // Prewitt
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = EdgeDetection::prewitt(currentImage);
                std::cout << "Prewitt edge detection applied.\n";
                break;
            }
            
            case 44: {
                // Laplacian
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = EdgeDetection::laplacian(currentImage);
                std::cout << "Laplacian edge detection applied.\n";
                break;
            }
            
            case 50:
            case 51:
            case 52:
            case 53: {
                // Morphological operations
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int size;
                std::cout << "Enter structuring element size (odd number, e.g., 5): ";
                std::cin >> size;
                auto kernel = Morphological::getStructuringElement(Morphological::StructuringElement::RECTANGLE, size);
                
                if (choice == 50) {
                    currentImage = Morphological::erode(currentImage, kernel);
                    std::cout << "Erosion applied.\n";
                } else if (choice == 51) {
                    currentImage = Morphological::dilate(currentImage, kernel);
                    std::cout << "Dilation applied.\n";
                } else if (choice == 52) {
                    currentImage = Morphological::opening(currentImage, kernel);
                    std::cout << "Opening applied.\n";
                } else {
                    currentImage = Morphological::closing(currentImage, kernel);
                    std::cout << "Closing applied.\n";
                }
                break;
            }
            
            case 54: {
                // Morphological Gradient
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int size;
                std::cout << "Enter structuring element size: ";
                std::cin >> size;
                auto kernel = Morphological::getStructuringElement(Morphological::StructuringElement::RECTANGLE, size);
                currentImage = Morphological::morphologicalGradient(currentImage, kernel);
                std::cout << "Morphological gradient applied.\n";
                break;
            }
            
            case 60: {
                // Rotate
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                double angle;
                std::cout << "Enter rotation angle in degrees: ";
                std::cin >> angle;
                currentImage = Geometric::rotate(currentImage, angle);
                std::cout << "Image rotated.\n";
                break;
            }
            
            case 61: {
                // Resize
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int width, height;
                std::cout << "Enter new width: ";
                std::cin >> width;
                std::cout << "Enter new height: ";
                std::cin >> height;
                currentImage = Geometric::resize(currentImage, width, height);
                std::cout << "Image resized.\n";
                break;
            }
            
            case 62: {
                // Translate
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                int dx, dy;
                std::cout << "Enter horizontal translation (pixels): ";
                std::cin >> dx;
                std::cout << "Enter vertical translation (pixels): ";
                std::cin >> dy;
                currentImage = Geometric::translate(currentImage, dx, dy);
                std::cout << "Image translated.\n";
                break;
            }
            
            case 63: {
                // Flip Horizontal
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = Geometric::flipHorizontal(currentImage);
                std::cout << "Image flipped horizontally.\n";
                break;
            }
            
            case 64: {
                // Flip Vertical
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = Geometric::flipVertical(currentImage);
                std::cout << "Image flipped vertically.\n";
                break;
            }
            
            case 70: {
                // Split Channels
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                auto channels = ColorOps::splitChannels(currentImage);
                std::cout << "Channels split. Saving...\n";
                for (size_t i = 0; i < channels.size(); ++i) {
                    std::string fname = "channel_" + std::to_string(i) + ".png";
                    channels[i].save(fname);
                    std::cout << "Saved " << fname << "\n";
                }
                break;
            }
            
            case 71: {
                // RGB to HSV
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                currentImage = ColorOps::rgbToHsv(currentImage);
                std::cout << "Converted to HSV.\n";
                break;
            }
            
            case 72: {
                // Adjust Hue
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float delta;
                std::cout << "Enter hue delta (-180 to 180): ";
                std::cin >> delta;
                currentImage = ColorOps::adjustHue(currentImage, delta);
                std::cout << "Hue adjusted.\n";
                break;
            }
            
            case 73: {
                // Adjust Saturation
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float factor;
                std::cout << "Enter saturation factor (0.0 to 2.0): ";
                std::cin >> factor;
                currentImage = ColorOps::adjustSaturation(currentImage, factor);
                std::cout << "Saturation adjusted.\n";
                break;
            }
            
            case 74: {
                // Adjust Value
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float factor;
                std::cout << "Enter value factor (0.0 to 2.0): ";
                std::cin >> factor;
                currentImage = ColorOps::adjustValue(currentImage, factor);
                std::cout << "Value adjusted.\n";
                break;
            }
            
            case 75: {
                // Color Balance
                if (!currentImage.isValid()) { std::cout << "No image loaded.\n"; break; }
                float r, g, b;
                std::cout << "Enter red factor (0.0 to 2.0): ";
                std::cin >> r;
                std::cout << "Enter green factor (0.0 to 2.0): ";
                std::cin >> g;
                std::cout << "Enter blue factor (0.0 to 2.0): ";
                std::cin >> b;
                currentImage = ColorOps::colorBalance(currentImage, r, g, b);
                std::cout << "Color balance adjusted.\n";
                break;
            }
            
            default:
                std::cout << "Invalid choice. Please try again.\n";
                break;
        }
    }
    
    return 0;
}
