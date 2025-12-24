#!/usr/bin/env python3
import sys

# Read the file as binary
with open('src/main.cpp', 'rb') as f:
    content = f.read()

# Try to decode as UTF-16 LE (which would have null bytes between chars)
# If that doesn't work, try UTF-8
try:
    # Check if it's UTF-16 LE (starts with BOM or has pattern of null bytes)
    if content.startswith(b'\xff\xfe'):  # UTF-16 LE BOM
        text = content.decode('utf-16-le')
    elif len(content) > 2 and content[1::2] == b'\x00' * (len(content) // 2):
        # Pattern of null bytes every other byte suggests UTF-16 LE
        text = content.decode('utf-16-le')
    else:
        # Try UTF-8
        text = content.decode('utf-8')
except UnicodeDecodeError:
    # If UTF-16 fails, try UTF-8
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        # Last resort: try to remove null bytes and decode as UTF-8
        content_clean = content.replace(b'\x00', b'')
        text = content_clean.decode('utf-8', errors='ignore')

# Write back as UTF-8 without BOM
with open('src/main.cpp', 'w', encoding='utf-8', newline='\n') as f:
    f.write(text)

print("File encoding fixed!")

