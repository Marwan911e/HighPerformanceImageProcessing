#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <vector>
#include <cstdint>

class Image {
public:
    // Constructors and destructor
    Image();
    Image(int width, int height, int channels);
    Image(const Image& other);
    Image& operator=(const Image& other);
    ~Image();

    // Load and save
    bool load(const std::string& filename);
    bool save(const std::string& filename, int quality = 90) const;

    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
    uint8_t* getData() { return data; }
    const uint8_t* getData() const { return data; }
    
    // Pixel access
    uint8_t* getPixel(int x, int y);
    const uint8_t* getPixel(int x, int y) const;
    void setPixel(int x, int y, const uint8_t* pixel);
    
    // Utility
    bool isValid() const { return data != nullptr && width > 0 && height > 0; }
    size_t getDataSize() const { return width * height * channels; }
    
    // Clone
    Image clone() const;
    
    // Create empty image with same dimensions
    Image createSimilar() const;

private:
    int width;
    int height;
    int channels;
    uint8_t* data;
    
    void allocate(int w, int h, int c);
    void deallocate();
};

#endif // IMAGE_H
