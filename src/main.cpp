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
#include <chrono>

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

// Note: autoSave function removed - operations now stay in memory until user saves (option 2)

void displayMenu() {
    std::cout << "\n=== IMAGE PROCESSING OPERATIONS ===\n";
    std::cout << "1.  Select Image from Input Folder\n";
    std::cout << "2.  Save Current Image\n";
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
    std::vector<std::string> appliedOperations;  // Track operations for filename generation
    
    // Create folders if they don't exist
    fs::create_directories(inputFolder);
    fs::create_directories(outputFolder);
    
    std::cout << "========================================\n";
    std::cout << "  IMAGE PROCESSING APPLICATION (SERIAL)\n";
    std::cout << "========================================\n";
    std::cout << "\n📁 Input folder: " << inputFolder << "/\n";
    std::cout << "📁 Output folder: " << outputFolder << "/\n";
    std::cout << "Supported formats: JPG, PNG, BMP, TGA\n";
    std::cout << "\n💡 Tip: Apply multiple operations, then save once (option 2)\n";
    
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
                        appliedOperations.clear(); // Reset operations when loading new image
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
                // Save with smart filename generation
                if (!currentImage.isValid()) {
                    std::cout << "✗ Error: No image loaded.\n";
                    break;
                }
                
                std::string filename;
                
                if (appliedOperations.empty()) {
                    // No operations applied, ask for custom name
                    std::cout << "Enter output filename (with extension): ";
                    std::cin >> filename;
                } else {
                    // Generate filename from operations
                    std::string baseName = getBaseName(fs::path(currentFilename).filename().string());
                    std::string ext = getExtension(fs::path(currentFilename).filename().string());
                    filename = outputFolder + "/" + baseName;
                    
                    // Add all operation names
                    for (const auto& op : appliedOperations) {
                        filename += "_" + op;
                    }
                    filename += ext;
                    
                    std::cout << "\n📝 Auto-generated filename: " << filename << "\n";
                    std::cout << "   Applied operations: ";
                    for (size_t i = 0; i < appliedOperations.size(); ++i) {
                        std::cout << appliedOperations[i];
                        if (i < appliedOperations.size() - 1) std::cout << " → ";
                    }
                    std::cout << "\n";
                    std::cout << "Use this name? (y/n, or enter 'c' for custom): ";
                    char choice;
                    std::cin >> choice;
                    if (choice != 'y' && choice != 'Y') {
                        if (choice == 'c' || choice == 'C') {
                            std::cout << "Enter custom filename: ";
                            std::cin >> filename;
                        } else {
                            // User entered 'n', ask for custom name
                            std::cout << "Enter custom filename: ";
                            std::cin >> filename;
                        }
                    }
                }
                
                if (currentImage.save(filename)) {
                    std::cout << "✓ Image saved successfully to: " << filename << "\n";
                    appliedOperations.clear(); // Reset after successful save
                } else {
                    std::cout << "✗ Error: Failed to save image.\n";
                }
                break;
            }
            
            case 10: {
                // Grayscale
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded. Please select an image first (option 1).\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::grayscale(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("grayscale");
                std::cout << "✓ Converted to grayscale in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 11: {
                // Brightness
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int delta;
                std::cout << "Enter brightness delta (-255 to 255): ";
                std::cin >> delta;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::adjustBrightness(currentImage, delta);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("brightness" + std::to_string(delta));
                std::cout << "✓ Brightness adjusted in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 12: {
                // Contrast
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float factor;
                std::cout << "Enter contrast factor (0.0 to 3.0): ";
                std::cin >> factor;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::adjustContrast(currentImage, factor);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string factorStr = std::to_string(factor);
                size_t dotPos = factorStr.find(".");
                if (dotPos != std::string::npos) {
                    factorStr = factorStr.substr(0, dotPos + 2);
                }
                appliedOperations.push_back("contrast" + factorStr);
                std::cout << "✓ Contrast adjusted in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 13: {
                // Manual Threshold
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int thresh;
                std::cout << "Enter threshold value (0-255): ";
                std::cin >> thresh;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::threshold(currentImage, thresh);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("threshold" + std::to_string(thresh));
                std::cout << "✓ Threshold applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 14: {
                // Otsu Threshold
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::thresholdOtsu(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("otsu");
                std::cout << "✓ Otsu threshold applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 15: {
                // Adaptive Threshold
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int blockSize;
                std::cout << "Enter block size (odd number, e.g., 11): ";
                std::cin >> blockSize;
                if (blockSize < 3 || blockSize % 2 == 0) {
                    std::cout << "⚠ Warning: Block size must be odd and >= 3. Using 11 instead.\n";
                    blockSize = 11;
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::adaptiveThreshold(currentImage, blockSize, 2, true);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("adaptthresh" + std::to_string(blockSize));
                std::cout << "✓ Adaptive threshold applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 16: {
                // Invert
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::invert(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("invert");
                std::cout << "✓ Image inverted in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 17: {
                // Gamma Correction
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float gamma;
                std::cout << "Enter gamma value (e.g., 0.5, 1.0, 2.2): ";
                std::cin >> gamma;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = PointOps::gammaCorrection(currentImage, gamma);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string gammaStr = std::to_string(gamma);
                size_t dotPos = gammaStr.find(".");
                if (dotPos != std::string::npos) {
                    gammaStr = gammaStr.substr(0, dotPos + 2);
                }
                appliedOperations.push_back("gamma" + gammaStr);
                std::cout << "✓ Gamma correction applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 20: {
                // Salt & Pepper
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float amount;
                std::cout << "Enter noise amount (0.0 to 1.0): ";
                std::cin >> amount;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Noise::saltAndPepper(currentImage, amount);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string amountStr = std::to_string(amount);
                size_t dotPos = amountStr.find(".");
                if (dotPos != std::string::npos) {
                    amountStr = amountStr.substr(0, dotPos + 2);
                }
                appliedOperations.push_back("saltpepper" + amountStr);
                std::cout << "✓ Salt & pepper noise added in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 21: {
                // Gaussian Noise
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float mean, stddev;
                std::cout << "Enter mean (e.g., 0): ";
                std::cin >> mean;
                std::cout << "Enter standard deviation (e.g., 25): ";
                std::cin >> stddev;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Noise::gaussian(currentImage, mean, stddev);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("gaussnoise" + std::to_string((int)stddev));
                std::cout << "✓ Gaussian noise added in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 22: {
                // Speckle Noise
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float variance;
                std::cout << "Enter variance (e.g., 0.1): ";
                std::cin >> variance;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Noise::speckle(currentImage, variance);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string varStr = std::to_string(variance);
                size_t dotPos = varStr.find(".");
                if (dotPos != std::string::npos) {
                    varStr = varStr.substr(0, dotPos + 2);
                }
                appliedOperations.push_back("speckle" + varStr);
                std::cout << "✓ Speckle noise added in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 30: {
                // Box Blur
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                if (size < 3 || size % 2 == 0) {
                    std::cout << "⚠ Warning: Kernel size must be odd and >= 3. Using 5 instead.\n";
                    size = 5;
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Filters::boxBlur(currentImage, size);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("boxblur" + std::to_string(size));
                std::cout << "✓ Box blur applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 31: {
                // Gaussian Blur
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                float sigma;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                if (size < 3 || size % 2 == 0) {
                    std::cout << "⚠ Warning: Kernel size must be odd and >= 3. Using 5 instead.\n";
                    size = 5;
                }
                std::cout << "Enter sigma (e.g., 1.4): ";
                std::cin >> sigma;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Filters::gaussianBlur(currentImage, size, sigma);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string sigmaStr = std::to_string(sigma);
                size_t dotPos = sigmaStr.find(".");
                if (dotPos != std::string::npos) {
                    sigmaStr = sigmaStr.substr(0, dotPos + 2);
                }
                appliedOperations.push_back("gaussblur" + std::to_string(size) + "s" + sigmaStr);
                std::cout << "✓ Gaussian blur applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 32: {
                // Median Filter
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter kernel size (odd number, e.g., 5): ";
                std::cin >> size;
                if (size < 3 || size % 2 == 0) {
                    std::cout << "⚠ Warning: Kernel size must be odd and >= 3. Using 5 instead.\n";
                    size = 5;
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Filters::medianFilter(currentImage, size);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("median" + std::to_string(size));
                std::cout << "✓ Median filter applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 33: {
                // Bilateral Filter
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
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Filters::bilateralFilter(currentImage, diameter, sigmaColor, sigmaSpace);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("bilateral" + std::to_string(diameter));
                std::cout << "✓ Bilateral filter applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 40: {
                // Sobel
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = EdgeDetection::sobel(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("sobel");
                std::cout << "✓ Sobel edge detection applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 41: {
                // Canny
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                double low, high;
                std::cout << "Enter low threshold (e.g., 50): ";
                std::cin >> low;
                std::cout << "Enter high threshold (e.g., 150): ";
                std::cin >> high;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = EdgeDetection::canny(currentImage, low, high);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("canny" + std::to_string((int)low) + "x" + std::to_string((int)high));
                std::cout << "✓ Canny edge detection applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 42: {
                // Sharpen
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = EdgeDetection::sharpen(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("sharpen");
                std::cout << "✓ Sharpen filter applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 43: {
                // Prewitt
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = EdgeDetection::prewitt(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("prewitt");
                std::cout << "✓ Prewitt edge detection applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 44: {
                // Laplacian
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = EdgeDetection::laplacian(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("laplacian");
                std::cout << "✓ Laplacian edge detection applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 50:
            case 51:
            case 52:
            case 53: {
                // Morphological operations
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter structuring element size (odd number, e.g., 5): ";
                std::cin >> size;
                if (size < 3 || size % 2 == 0) {
                    std::cout << "⚠ Warning: Size must be odd and >= 3. Using 5 instead.\n";
                    size = 5;
                }
                auto kernel = Morphological::getStructuringElement(Morphological::StructuringElement::RECTANGLE, size);
                
                auto start = std::chrono::high_resolution_clock::now();
                std::string opName;
                if (choice == 50) {
                    currentImage = Morphological::erode(currentImage, kernel);
                    opName = "erode" + std::to_string(size);
                } else if (choice == 51) {
                    currentImage = Morphological::dilate(currentImage, kernel);
                    opName = "dilate" + std::to_string(size);
                } else if (choice == 52) {
                    currentImage = Morphological::opening(currentImage, kernel);
                    opName = "open" + std::to_string(size);
                } else {
                    currentImage = Morphological::closing(currentImage, kernel);
                    opName = "close" + std::to_string(size);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                appliedOperations.push_back(opName);
                std::string opDesc = (choice == 50) ? "Erosion" : 
                                     (choice == 51) ? "Dilation" : 
                                     (choice == 52) ? "Opening" : "Closing";
                std::cout << "✓ " << opDesc << " applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 54: {
                // Morphological Gradient
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int size;
                std::cout << "Enter structuring element size: ";
                std::cin >> size;
                if (size < 3 || size % 2 == 0) {
                    std::cout << "⚠ Warning: Size must be odd and >= 3. Using 5 instead.\n";
                    size = 5;
                }
                auto kernel = Morphological::getStructuringElement(Morphological::StructuringElement::RECTANGLE, size);
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Morphological::morphologicalGradient(currentImage, kernel);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("morphgrad" + std::to_string(size));
                std::cout << "✓ Morphological gradient applied in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 60: {
                // Rotate
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                double angle;
                std::cout << "Enter rotation angle in degrees: ";
                std::cin >> angle;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Geometric::rotate(currentImage, angle);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("rotate" + std::to_string((int)angle));
                std::cout << "✓ Image rotated in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 61: {
                // Resize
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int width, height;
                std::cout << "Enter new width: ";
                std::cin >> width;
                std::cout << "Enter new height: ";
                std::cin >> height;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Geometric::resize(currentImage, width, height);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("resize" + std::to_string(width) + "x" + std::to_string(height));
                std::cout << "✓ Image resized in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 62: {
                // Translate
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                int dx, dy;
                std::cout << "Enter horizontal translation (pixels): ";
                std::cin >> dx;
                std::cout << "Enter vertical translation (pixels): ";
                std::cin >> dy;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Geometric::translate(currentImage, dx, dy);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("translate" + std::to_string(dx) + "x" + std::to_string(dy));
                std::cout << "✓ Image translated in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 63: {
                // Flip Horizontal
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Geometric::flipHorizontal(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("fliph");
                std::cout << "✓ Image flipped horizontally in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 64: {
                // Flip Vertical
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = Geometric::flipVertical(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("flipv");
                std::cout << "✓ Image flipped vertically in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 70: {
                // Split Channels (special case - saves multiple files)
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                auto channels = ColorOps::splitChannels(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::cout << "✓ Channels split in " << duration.count() 
                          << " ms (Serial). Saving...\n";
                for (size_t i = 0; i < channels.size(); ++i) {
                    std::string baseName = getBaseName(fs::path(currentFilename).filename().string());
                    std::string fname = outputFolder + "/" + baseName + "_channel_" + std::to_string(i) + ".png";
                    channels[i].save(fname);
                    std::cout << "  ✓ Saved " << fname << "\n";
                }
                break;
            }
            
            case 71: {
                // RGB to HSV
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = ColorOps::rgbToHsv(currentImage);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("hsv");
                std::cout << "✓ Converted to HSV in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 72: {
                // Adjust Hue
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float delta;
                std::cout << "Enter hue delta (-180 to 180): ";
                std::cin >> delta;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = ColorOps::adjustHue(currentImage, delta);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                appliedOperations.push_back("hue" + std::to_string((int)delta));
                std::cout << "✓ Hue adjusted in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 73: {
                // Adjust Saturation
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float factor;
                std::cout << "Enter saturation factor (0.0 to 2.0): ";
                std::cin >> factor;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = ColorOps::adjustSaturation(currentImage, factor);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string factorStr = std::to_string(factor);
                size_t dotPos = factorStr.find(".");
                if (dotPos != std::string::npos) {
                    factorStr = factorStr.substr(0, dotPos + 2);
                }
                appliedOperations.push_back("sat" + factorStr);
                std::cout << "✓ Saturation adjusted in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 74: {
                // Adjust Value
                if (!currentImage.isValid()) { 
                    std::cout << "✗ No image loaded.\n"; 
                    break; 
                }
                float factor;
                std::cout << "Enter value factor (0.0 to 2.0): ";
                std::cin >> factor;
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = ColorOps::adjustValue(currentImage, factor);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string factorStr = std::to_string(factor);
                size_t dotPos = factorStr.find(".");
                if (dotPos != std::string::npos) {
                    factorStr = factorStr.substr(0, dotPos + 2);
                }
                appliedOperations.push_back("val" + factorStr);
                std::cout << "✓ Value adjusted in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            case 75: {
                // Color Balance
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
                auto start = std::chrono::high_resolution_clock::now();
                currentImage = ColorOps::colorBalance(currentImage, r, g, b);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                std::string rStr = std::to_string(r);
                std::string gStr = std::to_string(g);
                std::string bStr = std::to_string(b);
                size_t dotPos = rStr.find(".");
                if (dotPos != std::string::npos) rStr = rStr.substr(0, dotPos + 1);
                dotPos = gStr.find(".");
                if (dotPos != std::string::npos) gStr = gStr.substr(0, dotPos + 1);
                dotPos = bStr.find(".");
                if (dotPos != std::string::npos) bStr = bStr.substr(0, dotPos + 1);
                appliedOperations.push_back("colbal" + rStr + "x" + gStr + "x" + bStr);
                std::cout << "✓ Color balance adjusted in " << duration.count() 
                          << " ms (Serial)\n";
                break;
            }
            
            default:
                std::cout << "✗ Invalid choice. Please try again.\n";
                break;
        }
    }
    
    return 0;
}
