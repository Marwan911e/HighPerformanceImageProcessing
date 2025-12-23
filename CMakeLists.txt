cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/lib)

# Source files
set(SOURCES
    src/main.cpp
    src/image.cpp
    src/point_operations.cpp
    src/noise.cpp
    src/filters.cpp
    src/edge_detection.cpp
    src/morphological.cpp
    src/geometric.cpp
    src/color_operations.cpp
)

# Create executable
add_executable(image_processor ${SOURCES})

# Platform-specific settings
if(WIN32)
    target_compile_definitions(image_processor PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Optional: Add OpenMP support for future parallel version
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(image_processor PUBLIC OpenMP::OpenMP_CXX)
endif()
