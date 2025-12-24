# Quick Start Guide

## 🎯 Fast Track to Using Your Image Processor

### For First-Time Users

#### Step 1: Choose Your Version

**Serial Version (Main Branch)** - For learning, debugging, or small images:
```bash
git checkout main
```

**OpenMP Version (OpenMP Branch)** - For production and large images:
```bash
git checkout openmp
```

#### Step 2: Build

```bash
make
```

Or if make is not available:
```bash
# Serial (main branch)
g++ -std=c++17 -O2 -I./include -I./lib src/*.cpp -o image_processor

# OpenMP (openmp branch)
g++ -std=c++17 -O2 -fopenmp -I./include -I./lib src/*.cpp -o image_processor -fopenmp
```

#### Step 3: Run

```bash
./image_processor    # Unix/Linux/Mac
image_processor.exe  # Windows
```

#### Step 4: Basic Usage

1. **Load image**: Enter `1`, then filename (e.g., `photo.jpg`)
2. **Apply operation**: Enter operation number (e.g., `10` for grayscale)
3. **Save result**: Enter `2`, then output filename (e.g., `output.jpg`)
4. **Exit**: Enter `0`

---

## 📚 Common Operations Quick Reference

### Enhance Photo
```
1 → Load image
11 → Brightness (+20)
12 → Contrast (1.3)
42 → Sharpen
2 → Save
```

### Edge Detection
```
1 → Load image
10 → Grayscale
31 → Gaussian Blur (5, 1.4)
40 → Sobel
2 → Save
```

### Remove Noise
```
1 → Load image
32 → Median Filter (5)
2 → Save
```

### Black & White
```
1 → Load image
10 → Grayscale
14 → Otsu Threshold
2 → Save
```

---

## ⚡ OpenMP Performance Tips

### Set Thread Count
```bash
# Linux/Mac
export OMP_NUM_THREADS=4
./image_processor

# Windows
$env:OMP_NUM_THREADS=4
.\image_processor.exe
```

### When to Use OpenMP
- ✅ Large images (>1024x1024)
- ✅ Batch processing
- ✅ Multi-core CPU available
- ✅ Performance is critical

### When to Use Serial
- ✅ Small images (<512x512)
- ✅ Debugging
- ✅ Learning the code
- ✅ Single-core system

---

## 🌿 Branch Management

### View All Branches
```bash
git branch -a
```

### Switch Branches
```bash
git checkout main     # Serial version
git checkout openmp   # Parallel version
```

### Compare Branches
```bash
git diff main openmp
```

### Pull Latest Changes
```bash
git pull origin main
git pull origin openmp
```

---

## 📁 Documentation Files

| File | Description |
|------|-------------|
| **README.md** | Complete project documentation |
| **OPENMP_IMPLEMENTATION.md** | OpenMP parallelization details |
| **BRANCH_COMPARISON.md** | Serial vs OpenMP comparison |
| **QUICK_START.md** | This file - quick reference |

---

## 🆘 Troubleshooting

### Can't Find OpenMP
```bash
# Install OpenMP (Ubuntu/Debian)
sudo apt-get install libomp-dev

# Or just use serial version
git checkout main
```

### Image Won't Load
- Check file format (JPG, PNG, BMP, TGA)
- Verify file path is correct
- Ensure file has read permissions

### Compilation Errors
```bash
# Clean and rebuild
make clean
make

# Or specify full command
g++ -std=c++17 -O2 -fopenmp -I./include -I./lib src/*.cpp -o image_processor -fopenmp
```

### Slow Performance
- Use OpenMP branch for large images
- Increase thread count: `export OMP_NUM_THREADS=8`
- Enable optimizations: `-O2` or `-O3`

---

## 🔗 Quick Links

- **GitHub Repo**: https://github.com/Marwan911e/HighPerformanceImageProcessing
- **Report Issues**: https://github.com/Marwan911e/HighPerformanceImageProcessing/issues
- **Main Branch**: https://github.com/Marwan911e/HighPerformanceImageProcessing/tree/main
- **OpenMP Branch**: https://github.com/Marwan911e/HighPerformanceImageProcessing/tree/openmp

---

## 📊 Operation Numbers Cheat Sheet

| Category | Numbers | Examples |
|----------|---------|----------|
| **File Operations** | 1-2 | Load (1), Save (2) |
| **Point Operations** | 10-17 | Grayscale (10), Brightness (11) |
| **Noise** | 20-22 | Salt&Pepper (20), Gaussian (21) |
| **Filters** | 30-33 | Box Blur (30), Gaussian (31) |
| **Edge Detection** | 40-44 | Sobel (40), Canny (41) |
| **Morphological** | 50-54 | Erosion (50), Dilation (51) |
| **Geometric** | 60-64 | Rotate (60), Resize (61) |
| **Color** | 70-75 | Split (70), HSV (71) |
| **Exit** | 0 | Exit program |

---

## 💡 Pro Tips

1. **Chain operations** - Apply multiple operations before saving
2. **Use tab completion** - For file paths (if your shell supports it)
3. **Batch process** - Create a commands text file and use: `./image_processor < commands.txt`
4. **Benchmark** - Time operations: `time ./image_processor`
5. **Compare versions** - Process same image on both branches to see speedup

---

## 🎓 Learning Path

### Beginner
1. Start with **main branch** (serial)
2. Try basic operations (grayscale, brightness, contrast)
3. Experiment with filters
4. Read the code to understand algorithms

### Intermediate
1. Switch to **openmp branch**
2. Compare performance with serial version
3. Study OpenMP directives in code
4. Experiment with thread counts

### Advanced
1. Modify parallelization strategies
2. Add new operations
3. Benchmark and optimize
4. Contribute improvements

---

## ✅ Verification Checklist

After building, verify everything works:

- [ ] Application starts without errors
- [ ] Can load a test image
- [ ] Grayscale operation works (option 10)
- [ ] Can save output image
- [ ] Output image looks correct
- [ ] Performance is acceptable
- [ ] Can exit cleanly (option 0)

---

## 📞 Need More Help?

- **Full Documentation**: See `README.md`
- **OpenMP Details**: See `OPENMP_IMPLEMENTATION.md`
- **Performance Comparison**: See `BRANCH_COMPARISON.md`
- **GitHub Issues**: Report problems or ask questions

---

**Happy Image Processing! 🎨**

