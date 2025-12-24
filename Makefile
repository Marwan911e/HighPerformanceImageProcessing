# Makefile for Image Processing Project (OpenMP Version)
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -fopenmp -I./include -I./lib
LDFLAGS = -fopenmp 

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

# Target executable (in build directory)
TARGET = $(OBJDIR)/image_processor

# Default target
all: $(TARGET)

# Create build directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Link
$(TARGET): $(OBJECTS) | $(OBJDIR)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJDIR)
	rm -f image_processor

# Rebuild
rebuild: clean all

.PHONY: all clean rebuild
