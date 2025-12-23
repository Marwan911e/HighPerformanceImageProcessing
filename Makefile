# Makefile for Image Processing Project
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -I./include -I./lib
LDFLAGS = 

# Source files
SRCDIR = src
SOURCES = $(SRCDIR)/main.cpp \
          $(SRCDIR)/image.cpp \
          $(SRCDIR)/point_operations.cpp \
          $(SRCDIR)/noise.cpp \
          $(SRCDIR)/filters.cpp \
          $(SRCDIR)/edge_detection.cpp \
          $(SRCDIR)/morphological.cpp \
          $(SRCDIR)/geometric.cpp \
          $(SRCDIR)/color_operations.cpp

# Object files
OBJDIR = build
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

# Target executable
TARGET = image_processor

# Default target
all: $(TARGET)

# Create build directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Rebuild
rebuild: clean all

.PHONY: all clean rebuild
