#include "color_operations.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ColorOps {

std::vector<Image> splitChannels(const Image& img) {
    std::vector<Image> channels;
    if (!img.isValid()) return channels;
    
    for (int c = 0; c < img.getChannels(); ++c) {
        Image channel(img.getWidth(), img.getHeight(), 1);
        
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < img.getHeight(); ++y) {
            for (int x = 0; x < img.getWidth(); ++x) {
                const uint8_t* pixel = img.getPixel(x, y);
                channel.setPixel(x, y, &pixel[c]);
            }
        }
        
        channels.push_back(channel);
    }
    
    return channels;
}

Image mergeChannels(const std::vector<Image>& channels) {
    if (channels.empty() || !channels[0].isValid()) return Image();
    
    int width = channels[0].getWidth();
    int height = channels[0].getHeight();
    int numChannels = channels.size();
    
    Image result(width, height, numChannels);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t pixel[4] = {0, 0, 0, 255};
            
            for (int c = 0; c < numChannels && c < 4; ++c) {
                const uint8_t* ch = channels[c].getPixel(x, y);
                pixel[c] = *ch;
            }
            
            result.setPixel(x, y, pixel);
        }
    }
    
    return result;
}

Image rgbToHsv(const Image& img) {
    if (!img.isValid() || img.getChannels() < 3) return Image();
    
    Image result(img.getWidth(), img.getHeight(), 3);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            const uint8_t* rgb = img.getPixel(x, y);
            
            float r = rgb[0] / 255.0f;
            float g = rgb[1] / 255.0f;
            float b = rgb[2] / 255.0f;
            
            float max = std::max({r, g, b});
            float min = std::min({r, g, b});
            float delta = max - min;
            
            float h = 0, s = 0, v = max;
            
            if (delta > 0.00001f) {
                s = delta / max;
                
                if (max == r) {
                    h = 60.0f * (std::fmod((g - b) / delta, 6.0f));
                } else if (max == g) {
                    h = 60.0f * ((b - r) / delta + 2.0f);
                } else {
                    h = 60.0f * ((r - g) / delta + 4.0f);
                }
                
                if (h < 0) h += 360.0f;
            }
            
            uint8_t hsv[3];
            hsv[0] = static_cast<uint8_t>(h / 2.0f);  // H: 0-179
            hsv[1] = static_cast<uint8_t>(s * 255.0f); // S: 0-255
            hsv[2] = static_cast<uint8_t>(v * 255.0f); // V: 0-255
            
            result.setPixel(x, y, hsv);
        }
    }
    
    return result;
}

Image hsvToRgb(const Image& img) {
    if (!img.isValid() || img.getChannels() < 3) return Image();
    
    Image result(img.getWidth(), img.getHeight(), 3);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < img.getHeight(); ++y) {
        for (int x = 0; x < img.getWidth(); ++x) {
            const uint8_t* hsv = img.getPixel(x, y);
            
            float h = hsv[0] * 2.0f;
            float s = hsv[1] / 255.0f;
            float v = hsv[2] / 255.0f;
            
            float c = v * s;
            float x_val = c * (1 - std::abs(std::fmod(h / 60.0f, 2.0f) - 1));
            float m = v - c;
            
            float r = 0, g = 0, b = 0;
            
            if (h < 60) {
                r = c; g = x_val; b = 0;
            } else if (h < 120) {
                r = x_val; g = c; b = 0;
            } else if (h < 180) {
                r = 0; g = c; b = x_val;
            } else if (h < 240) {
                r = 0; g = x_val; b = c;
            } else if (h < 300) {
                r = x_val; g = 0; b = c;
            } else {
                r = c; g = 0; b = x_val;
            }
            
            uint8_t rgb[3];
            rgb[0] = static_cast<uint8_t>((r + m) * 255.0f);
            rgb[1] = static_cast<uint8_t>((g + m) * 255.0f);
            rgb[2] = static_cast<uint8_t>((b + m) * 255.0f);
            
            result.setPixel(x, y, rgb);
        }
    }
    
    return result;
}

Image adjustHue(const Image& img, float delta) {
    if (!img.isValid() || img.getChannels() < 3) return Image();
    
    Image hsv = rgbToHsv(img);
    
    for (int y = 0; y < hsv.getHeight(); ++y) {
        for (int x = 0; x < hsv.getWidth(); ++x) {
            uint8_t* pixel = hsv.getPixel(x, y);
            float h = pixel[0] * 2.0f + delta;
            while (h < 0) h += 360.0f;
            while (h >= 360.0f) h -= 360.0f;
            pixel[0] = static_cast<uint8_t>(h / 2.0f);
        }
    }
    
    return hsvToRgb(hsv);
}

Image adjustSaturation(const Image& img, float factor) {
    if (!img.isValid() || img.getChannels() < 3) return Image();
    
    Image hsv = rgbToHsv(img);
    
    for (int y = 0; y < hsv.getHeight(); ++y) {
        for (int x = 0; x < hsv.getWidth(); ++x) {
            uint8_t* pixel = hsv.getPixel(x, y);
            float s = pixel[1] * factor;
            pixel[1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, s)));
        }
    }
    
    return hsvToRgb(hsv);
}

Image adjustValue(const Image& img, float factor) {
    if (!img.isValid() || img.getChannels() < 3) return Image();
    
    Image hsv = rgbToHsv(img);
    
    for (int y = 0; y < hsv.getHeight(); ++y) {
        for (int x = 0; x < hsv.getWidth(); ++x) {
            uint8_t* pixel = hsv.getPixel(x, y);
            float v = pixel[2] * factor;
            pixel[2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, v)));
        }
    }
    
    return hsvToRgb(hsv);
}

Image rgbToLab(const Image& img) {
    // Placeholder - full LAB conversion requires complex color space math
    return img.clone();
}

Image labToRgb(const Image& img) {
    // Placeholder - full LAB conversion requires complex color space math
    return img.clone();
}

Image colorBalance(const Image& img, float redFactor, float greenFactor, float blueFactor) {
    if (!img.isValid() || img.getChannels() < 3) return Image();
    
    Image result = img.clone();
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < result.getHeight(); ++y) {
        for (int x = 0; x < result.getWidth(); ++x) {
            uint8_t* pixel = result.getPixel(x, y);
            pixel[0] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, pixel[0] * redFactor)));
            pixel[1] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, pixel[1] * greenFactor)));
            pixel[2] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, pixel[2] * blueFactor)));
        }
    }
    
    return result;
}

Image toneMapping(const Image& img, float exposure, float gamma) {
    if (!img.isValid()) return Image();
    
    Image result = img.clone();
    uint8_t* data = result.getData();
    size_t size = result.getDataSize();
    
    // Apply exposure
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        float val = data[i] / 255.0f;
        val = val * std::pow(2.0f, exposure);
        val = std::pow(val, 1.0f / gamma);
        data[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val * 255.0f)));
    }
    
    return result;
}

} // namespace ColorOps
