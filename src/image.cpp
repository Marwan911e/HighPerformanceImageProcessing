#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"
#include "image.h"
#include <cstring>
#include <algorithm>

Image::Image() : width(0), height(0), channels(0), data(nullptr) {}

Image::Image(int width, int height, int channels) 
    : width(0), height(0), channels(0), data(nullptr) {
    allocate(width, height, channels);
}

Image::Image(const Image& other) 
    : width(0), height(0), channels(0), data(nullptr) {
    if (other.isValid()) {
        allocate(other.width, other.height, other.channels);
        std::memcpy(data, other.data, getDataSize());
    }
}

Image& Image::operator=(const Image& other) {
    if (this != &other) {
        deallocate();
        if (other.isValid()) {
            allocate(other.width, other.height, other.channels);
            std::memcpy(data, other.data, getDataSize());
        }
    }
    return *this;
}

Image::~Image() {
    deallocate();
}

bool Image::load(const std::string& filename) {
    deallocate();
    
    int w, h, c;
    uint8_t* img_data = stbi_load(filename.c_str(), &w, &h, &c, 0);
    
    if (!img_data) {
        return false;
    }
    
    allocate(w, h, c);
    std::memcpy(data, img_data, getDataSize());
    stbi_image_free(img_data);
    
    return true;
}

bool Image::save(const std::string& filename, int quality) const {
    if (!isValid()) {
        return false;
    }
    
    std::string ext = filename.substr(filename.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "png") {
        return stbi_write_png(filename.c_str(), width, height, channels, data, width * channels) != 0;
    } else if (ext == "jpg" || ext == "jpeg") {
        return stbi_write_jpg(filename.c_str(), width, height, channels, data, quality) != 0;
    } else if (ext == "bmp") {
        return stbi_write_bmp(filename.c_str(), width, height, channels, data) != 0;
    } else if (ext == "tga") {
        return stbi_write_tga(filename.c_str(), width, height, channels, data) != 0;
    }
    
    return false;
}

uint8_t* Image::getPixel(int x, int y) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return nullptr;
    }
    return &data[(y * width + x) * channels];
}

const uint8_t* Image::getPixel(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return nullptr;
    }
    return &data[(y * width + x) * channels];
}

void Image::setPixel(int x, int y, const uint8_t* pixel) {
    if (x >= 0 && x < width && y >= 0 && y < height && pixel) {
        std::memcpy(&data[(y * width + x) * channels], pixel, channels);
    }
}

Image Image::clone() const {
    return Image(*this);
}

Image Image::createSimilar() const {
    return Image(width, height, channels);
}

void Image::allocate(int w, int h, int c) {
    if (w > 0 && h > 0 && c > 0) {
        width = w;
        height = h;
        channels = c;
        data = new uint8_t[w * h * c];
        std::memset(data, 0, w * h * c);
    }
}

void Image::deallocate() {
    if (data) {
        delete[] data;
        data = nullptr;
    }
    width = height = channels = 0;
}
