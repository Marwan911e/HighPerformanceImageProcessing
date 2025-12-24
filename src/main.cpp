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
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

// Helper function to list images in a directory
std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> images;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return images;
    }
    
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
                ext == ".bmp" || ext == ".tga") {
                images.push_back(entry.path().filename().string());
            }
        }
    }
    std::sort(images.begin(), images.end()); // Sort alphabetically
    return images;
}

// Helper function to get base filename without extension
std::string getBaseName(const std::string& filename) {
    fs::path p(filename);
    std::string stem = p.stem().string();
    return stem;
}

// Helper function to get extension
std::string getExtension(const std::string& filename) {
    fs::path p(filename);
    return p.extension().string();
}

// Auto-save function
void autoSave(const Image& img, const std::string& inputPath, 
              const std::string& operationName, const std::string& outputFolder) {
    std::string baseName = getBaseName(fs::path(inputPath).filename().string());
    std::string ext = getExtension(fs::path(inputPath).filename().string());
    std::string outputPath = outputFolder + "/" + baseName + "_" + operationName + ext;
    
    if (img.save(outputPath)) {
        std::cout << "✓ Saved to: " << outputPath << "\n";
    } else {
        std::cout << "✗ Error: Failed to save image.\n";
    }
}

void displayMenu() {
    std::cout << "\n=== IMAGE PROCESSING OPERATIONS ===\n";
    std::cout << "1.  Select Image from Input Folder\n";
    std::cout << "2.  Save Current Image (Manual)\n";
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
    std::string currentBaseName;
    std::string currentExtension;
    std::string inputFolder = "input";
    std::string outputFolder = "output";
    
    // Create folders if they don't exist
    fs::create_directories(inputFolder);
    fs::create_directories(outputFolder);
    
    std::cout << "========================================\n";
    std::cout << "  IMAGE PROCESSING APPLICATION (SERIAL)\n";
    std::cout << "========================================\n";
    std::cout << "\n📁 Input folder: " << inputFolder << "/\n";
    std::cout << "📁 Output folder: " << outputFolder << "/\n";
    std::cout << "Supported formats: JPG, PNG, BMP, TGA\n";
    
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
                // AUTOMATIC FILE BROWSER - Show all images and let user pick by number
                std::vector<std::string> images = listImages(inputFolder);
                
                if (images.empty()) {
                    std::cout << "\n⚠ No images found in " << inputFolder << "/ folder.\n";
                    std::cout << "Please place images in the " << inputFolder << "/ directory.\n";
                    break;
                }
                
                std::cout << "\n📋 Available images in " << inputFolder << "/:\n";
                std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                for (size_t i = 0; i < images.size(); ++i) {
                    // Show file size for better info
                    std::string fullPath = inputFolder + "/" + images[i];
                    if (fs::exists(fullPath)) {
                        auto fileSize = fs::file_size(fullPath);
                        double sizeMB = fileSize / (1024.0 * 1024.0);
                        std::cout << "  " << std::setw(2) << (i + 1) << ". " 
                                  << std::setw(30) << std::left << images[i]
                                  << " (" << std::fixed << std::setprecision(2) 
                                  << sizeMB << " MB)\n";
                    } else {
                        std::cout << "  " << std::setw(2) << (i + 1) << ". " 
                                  << images[i] << "\n";
                    }
                }
                std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                std::cout << "\nSelect image number (1-" << images.size() << "): ";
                
                int imgChoice;
                std::cin >> imgChoice;
                
                if (imgChoice >= 1 && imgChoice <= static_cast<int>(images.size())) {
                    currentFilename = inputFolder + "/" + images[imgChoice - 1];
                    if (currentImage.load(currentFilename)) {
                        currentBaseName = getBaseName(images[imgChoice - 1]);
                        currentExtension = getExtension(images[imgChoice - 1]);
                        std::cout << "\n✓ Image loaded: " << images[imgChoice - 1] << "\n";
                        std::cout << "  Dimensions: " << currentImage.getWidth() 
                                  << "x" << currentImage.getHeight();
                        std::cout << ", Channels: " << currentImage.getChannels() << "\n";
                    } else {
                        std::cout << "✗ Error: Failed to load image.\n";
                    }
                } else {
                    std::cout << "✗ Invalid selection. Please choose 1-" << images.size() << ".\n";
                }
                break;
            }
            
            case 2: {
                // Manual save option
                if (!currentImage.isValid()) {
                    std::cout << "✗ Error: No image loaded.\n";
                    break;
                }
                std::string filename;
                std::cout << "Enter output filename (with extension): ";
                std::cin >> filename;
                if (currentImage.save(filename)) {
                    std::cout << "✓ Image saved successfully!\n";
                } else {
                    std::cout << "✗ Error: Failed to save image.\n";
                }
                break;
            }
            
            case 10: {
                // Grayscale - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded. Please select an image first (option 1).\n"; 
                    break; 
                }
                currentImage = PointOps::grayscale(currentImage);
                std::cout << "✓ Converted to grayscale.\n";
                autoSave(currentImage, currentFilename, "grayscale", outputFolder);
                break;
            }
            
            case 11: {
                // Brightness - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int delta;
                std::cout << "Enter brightness delta (-255 to 255): ";
                std::cin >> delta;
                currentImage = PointOps::adjustBrightness(currentImage, delta);
                std::cout << "✓ Brightness adjusted.\n";
                autoSave(currentImage, currentFilename, "brightness_" + std::to_string(delta), outputFolder);
                break;
            }
            
            case 12: {
                // Contrast - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float factor;
                std::cout << "Enter contrast factor (0.0 to 3.0): ";
                std::cin >> factor;
                currentImage = PointOps::adjustContrast(currentImage, factor);
                std::cout << "✓ Contrast adjusted.\n";
                std::string factorStr = std::to_string(factor);
                size_t dotPos = factorStr.find(".");
                if (dotPos != std::string::npos) {
                    factorStr = factorStr.substr(0, dotPos + 2);
                }
                autoSave(currentImage, currentFilename, "contrast_" + factorStr, outputFolder);
                break;
            }
            
            case 13: {
                // Manual Threshold - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int thresh;
                std::cout << "Enter threshold value (0-255): ";
                std::cin >> thresh;
                currentImage = PointOps::threshold(currentImage, thresh);
                std::cout << "✓ Threshold applied.\n";
                autoSave(currentImage, currentFilename, "threshold_" + std::to_string(thresh), outputFolder);
                break;
            }
            
            case 14: {
                // Otsu Threshold - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = PointOps::thresholdOtsu(currentImage);
                std::cout << "✓ Otsu threshold applied.\n";
                autoSave(currentImage, currentFilename, "otsu_threshold", outputFolder);
                break;
            }
            
            case 15: {
                // Adaptive Threshold - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int blockSize;
                std::cout << "Enter block size (odd number, e.g., 11): ";
                std::cin >> blockSize;
                currentImage = PointOps::adaptiveThreshold(currentImage, blockSize, 2, true);
                std::cout << "✓ Adaptive threshold applied.\n";
                autoSave(currentImage, currentFilename, "adaptive_threshold_" + std::to_string(blockSize), outputFolder);
                break;
            }
            
            case 16: {
                // Invert - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = PointOps::invert(currentImage);
                std::cout << "✓ Image inverted.\n";
                autoSave(currentImage, currentFilename, "inverted", outputFolder);
                break;
            }
            
            case 17: {
                // Gamma Correction - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float gamma;
                std::cout << "Enter gamma value (e.g., 0.5, 1.0, 2.2): ";
                std::cin >> gamma;
                currentImage = PointOps::gammaCorrection(currentImage, gamma);
                std::cout << "✓ Gamma correction applied.\n";
                std::string gammaStr = std::to_string(gamma);
                size_t dotPos = gammaStr.find(".");
                if (dotPos != std::string::npos) {
                    gammaStr = gammaStr.substr(0, dotPos + 2);
                }
                autoSave(currentImage, currentFilename, "gamma_" + gammaStr, outputFolder);
                break;
            }
            
            case 20: {
                // Salt & Pepper - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float amount;
                std::cout << "Enter noise amount (0.0 to 1.0): ";
                std::cin >> amount;
                currentImage = Noise::saltAndPepper(currentImage, amount);
                std::cout << "✓ Salt & pepper noise added.\n";
                std::string amountStr = std::to_string(amount);
                size_t dotPos = amountStr.find(".");
                if (dotPos != std::string::npos) {
                    amountStr = amountStr.substr(0, dotPos + 2);
                }
                autoSave(currentImage, currentFilename, "saltpepper_" + amountStr, outputFolder);
                break;
            }
            
            case 21: {
                // Gaussian Noise - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float mean, stddev;
                std::cout << "Enter mean (e.g., 0): ";
                std::cin >> mean;
                std::cout << "Enter standard deviation (e.g., 25): ";
                std::cin >> stddev;
                currentImage = Noise::gaussian(currentImage, mean, stddev);
                std::cout << "✓ Gaussian noise added.\n";
                autoSave(currentImage, currentFilename, "gaussian_noise_" + std::to_string((int)stddev), outputFolder);
                break;
            }
            
            case 22: {
                // Speckle Noise - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float variance;
                std::cout << "Enter variance (e.g., 0.1): ";
                std::cin >> variance;
                currentImage = Noise::speckle(currentImage, variance);
                std::cout << "✓ Speckle noise added.\n";
                std::string varStr = std::to_string(variance);
                size_t dotPos = varStr.find(".");
                if (dotPos != std::string::npos) {
                    varStr = varStr.substr(0, dotPos + 2);
                }
                autoSave(currentImage, currentFilename, "speckle_" + varStr, outputFolder);
                break;
            }
            
            case 30: {
                // Box Blur - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                currentImage = Filters::boxBlur(currentImage, size);
                std::cout << "✓ Box blur applied.\n";
                autoSave(currentImage, currentFilename, "boxblur_" + std::to_string(size), outputFolder);
                break;
            }
            
            case 31: {
                // Gaussian Blur - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                float sigma;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                std::cout << "Enter sigma (e.g., 1.4): ";
                std::cin >> sigma;
                currentImage = Filters::gaussianBlur(currentImage, size, sigma);
                std::cout << "✓ Gaussian blur applied.\n";
                std::string sigmaStr = std::to_string(sigma);
                size_t dotPos = sigmaStr.find(".");
                if (dotPos != std::string::npos) {
                    sigmaStr = sigmaStr.substr(0, dotPos + 2);
                }
                autoSave(currentImage, currentFilename, "gaussianblur_" + std::to_string(size) + "_" + sigmaStr, outputFolder);
                break;
            }
            
            case 32: {
                // Median Filter - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                currentImage = Filters::medianFilter(currentImage, size);
                std::cout << "✓ Median filter applied.\n";
                autoSave(currentImage, currentFilename, "median_" + std::to_string(size), outputFolder);
                break;
            }
            
            case 33: {
                // Bilateral Filter - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int diameter;
                double sigmaColor, sigmaSpace;
                std::cout << "Enter diameter (e.g., 9): ";
                std::cin >> diameter;
                std::cout << "Enter sigma color (e.g., 75): ";
                std::cin >> sigmaColor;
                std::cout << "Enter sigma space (e.g., 75): ";
                std::cin >> sigmaSpace;
                currentImage = Filters::bilateralFilter(currentImage, diameter, sigmaColor, sigmaSpace);
                std::cout << "✓ Bilateral filter applied.\n";
                autoSave(currentImage, currentFilename, "bilateral_" + std::to_string(diameter), outputFolder);
                break;
            }
            
            case 40: {
                // Sobel - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = EdgeDetection::sobel(currentImage);
                std::cout << "✓ Sobel edge detection applied.\n";
                autoSave(currentImage, currentFilename, "sobel", outputFolder);
                break;
            }
            
            case 41: {
                // Canny - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                double low, high;
                std::cout << "Enter low threshold (e.g., 50): ";
                std::cin >> low;
                std::cout << "Enter high threshold (e.g., 150): ";
                std::cin >> high;
                currentImage = EdgeDetection::canny(currentImage, low, high);
                std::cout << "✓ Canny edge detection applied.\n";
                autoSave(currentImage, currentFilename, "canny_" + std::to_string((int)low) + "_" + std::to_string((int)high), outputFolder);
                break;
            }
            
            case 42: {
                // Sharpen - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = EdgeDetection::sharpen(currentImage);
                std::cout << "✓ Sharpen filter applied.\n";
                autoSave(currentImage, currentFilename, "sharpen", outputFolder);
                break;
            }
            
            case 43: {
                // Prewitt - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = EdgeDetection::prewitt(currentImage);
                std::cout << "✓ Prewitt edge detection applied.\n";
                autoSave(currentImage, currentFilename, "prewitt", outputFolder);
                break;
            }
            
            case 44: {
                // Laplacian - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = EdgeDetection::laplacian(currentImage);
                std::cout << "✓ Laplacian edge detection applied.\n";
                autoSave(currentImage, currentFilename, "laplacian", outputFolder);
                break;
            }
            
            case 50:
            case 51:
            case 52:
            case 53: {
                // Morphological operations - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter structuring element size (odd number, e.g., 5): ";
                std::cin >> size;
                auto kernel = Morphological::getStructuringElement(Morphological::StructuringElement::RECTANGLE, size);
                
                std::string opName;
                if (choice == 50) {
                    currentImage = Morphological::erode(currentImage, kernel);
                    std::cout << "✓ Erosion applied.\n";
                    opName = "erosion_" + std::to_string(size);
                } else if (choice == 51) {
                    currentImage = Morphological::dilate(currentImage, kernel);
                    std::cout << "✓ Dilation applied.\n";
                    opName = "dilation_" + std::to_string(size);
                } else if (choice == 52) {
                    currentImage = Morphological::opening(currentImage, kernel);
                    std::cout << "✓ Opening applied.\n";
                    opName = "opening_" + std::to_string(size);
                } else {
                    currentImage = Morphological::closing(currentImage, kernel);
                    std::cout << "✓ Closing applied.\n";
                    opName = "closing_" + std::to_string(size);
                }
                autoSave(currentImage, currentFilename, opName, outputFolder);
                break;
            }
            
            case 54: {
                // Morphological Gradient - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter structuring element size: ";
                std::cin >> size;
                auto kernel = Morphological::getStructuringElement(Morphological::StructuringElement::RECTANGLE, size);
                currentImage = Morphological::morphologicalGradient(currentImage, kernel);
                std::cout << "✓ Morphological gradient applied.\n";
                autoSave(currentImage, currentFilename, "morph_gradient_" + std::to_string(size), outputFolder);
                break;
            }
            
            case 60: {
                // Rotate - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                double angle;
                std::cout << "Enter rotation angle in degrees: ";
                std::cin >> angle;
                currentImage = Geometric::rotate(currentImage, angle);
                std::cout << "✓ Image rotated.\n";
                autoSave(currentImage, currentFilename, "rotate_" + std::to_string((int)angle), outputFolder);
                break;
            }
            
            case 61: {
                // Resize - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int width, height;
                std::cout << "Enter new width: ";
                std::cin >> width;
                std::cout << "Enter new height: ";
                std::cin >> height;
                currentImage = Geometric::resize(currentImage, width, height);
                std::cout << "✓ Image resized.\n";
                autoSave(currentImage, currentFilename, "resize_" + std::to_string(width) + "x" + std::to_string(height), outputFolder);
                break;
            }
            
            case 62: {
                // Translate - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int dx, dy;
                std::cout << "Enter horizontal translation (pixels): ";
                std::cin >> dx;
                std::cout << "Enter vertical translation (pixels): ";
                std::cin >> dy;
                currentImage = Geometric::translate(currentImage, dx, dy);
                std::cout << "✓ Image translated.\n";
                autoSave(currentImage, currentFilename, "translate_" + std::to_string(dx) + "_" + std::to_string(dy), outputFolder);
                break;
            }
            
            case 63: {
                // Flip Horizontal - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = Geometric::flipHorizontal(currentImage);
                std::cout << "✓ Image flipped horizontally.\n";
                autoSave(currentImage, currentFilename, "flip_horizontal", outputFolder);
                break;
            }
            
            case 64: {
                // Flip Vertical - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = Geometric::flipVertical(currentImage);
                std::cout << "✓ Image flipped vertically.\n";
                autoSave(currentImage, currentFilename, "flip_vertical", outputFolder);
                break;
            }
            
            case 70: {
                // Split Channels - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto channels = ColorOps::splitChannels(currentImage);
                std::cout << "✓ Channels split. Saving...\n";
                for (size_t i = 0; i < channels.size(); ++i) {
                    std::string baseName = getBaseName(currentFilename);
                    std::string fname = outputFolder + "/" + baseName + "_channel_" + std::to_string(i) + ".png";
                    channels[i].save(fname);
                    std::cout << "  ✓ Saved " << fname << "\n";
                }
                break;
            }
            
            case 71: {
                // RGB to HSV - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                currentImage = ColorOps::rgbToHsv(currentImage);
                std::cout << "✓ Converted to HSV.\n";
                autoSave(currentImage, currentFilename, "hsv", outputFolder);
                break;
            }
            
            case 72: {
                // Adjust Hue - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float delta;
                std::cout << "Enter hue delta (-180 to 180): ";
                std::cin >> delta;
                currentImage = ColorOps::adjustHue(currentImage, delta);
                std::cout << "✓ Hue adjusted.\n";
                autoSave(currentImage, currentFilename, "hue_" + std::to_string((int)delta), outputFolder);
                break;
            }
            
            case 73: {
                // Adjust Saturation - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float factor;
                std::cout << "Enter saturation factor (0.0 to 2.0): ";
                std::cin >> factor;
                currentImage = ColorOps::adjustSaturation(currentImage, factor);
                std::cout << "✓ Saturation adjusted.\n";
                std::string factorStr = std::to_string(factor);
                size_t dotPos = factorStr.find(".");
                if (dotPos != std::string::npos) {
                    factorStr = factorStr.substr(0, dotPos + 2);
                }
                autoSave(currentImage, currentFilename, "saturation_" + factorStr, outputFolder);
                break;
            }
            
            case 74: {
                // Adjust Value - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float factor;
                std::cout << "Enter value factor (0.0 to 2.0): ";
                std::cin >> factor;
                currentImage = ColorOps::adjustValue(currentImage, factor);
                std::cout << "✓ Value adjusted.\n";
                std::string factorStr = std::to_string(factor);
                size_t dotPos = factorStr.find(".");
                if (dotPos != std::string::npos) {
                    factorStr = factorStr.substr(0, dotPos + 2);
                }
                autoSave(currentImage, currentFilename, "value_" + factorStr, outputFolder);
                break;
            }
            
            case 75: {
                // Color Balance - with auto-save
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float r, g, b;
                std::cout << "Enter red factor (0.0 to 2.0): ";
                std::cin >> r;
                std::cout << "Enter green factor (0.0 to 2.0): ";
                std::cin >> g;
                std::cout << "Enter blue factor (0.0 to 2.0): ";
                std::cin >> b;
                currentImage = ColorOps::colorBalance(currentImage, r, g, b);
                std::cout << "✓ Color balance adjusted.\n";
                std::string rStr = std::to_string(r);
                std::string gStr = std::to_string(g);
                std::string bStr = std::to_string(b);
                size_t dotPos = rStr.find(".");
                if (dotPos != std::string::npos) rStr = rStr.substr(0, dotPos + 1);
                dotPos = gStr.find(".");
                if (dotPos != std::string::npos) gStr = gStr.substr(0, dotPos + 1);
                dotPos = bStr.find(".");
                if (dotPos != std::string::npos) bStr = bStr.substr(0, dotPos + 1);
                autoSave(currentImage, currentFilename, "colorbalance_" + rStr + "_" + gStr + "_" + bStr, outputFolder);
                break;
            }
            
            default:
                std::cout << "✗ Invalid choice. Please try again.\n";
                break;
        }
    }
    
    return 0;
}
