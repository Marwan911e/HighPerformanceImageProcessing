# Download Instructions for STB Libraries

## Quick Download (Windows PowerShell)

Run these commands in the project root directory:

```powershell
# Create lib directory if it doesn't exist
New-Item -ItemType Directory -Force -Path lib

# Download stb_image.h
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -OutFile "lib\stb_image.h"

# Download stb_image_write.h
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" -OutFile "lib\stb_image_write.h"

Write-Host "STB libraries downloaded successfully!" -ForegroundColor Green
```

## Quick Download (Linux/Mac)

```bash
# Download stb_image.h
curl -o lib/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h

# Download stb_image_write.h
curl -o lib/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

echo "STB libraries downloaded successfully!"
```

## Manual Download

If the above doesn't work, manually download these files:

1. Go to: https://github.com/nothings/stb
2. Download these two files:
   - `stb_image.h` - https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
   - `stb_image_write.h` - https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
3. Place both files in the `lib/` directory of this project

## Verification

After downloading, your `lib/` directory should contain:
- `stb_image.h` (~300 KB)
- `stb_image_write.h` (~60 KB)

You can verify by running:

**Windows:**
```powershell
dir lib
```

**Linux/Mac:**
```bash
ls -lh lib/
```

Both files should be present before compiling the project.
