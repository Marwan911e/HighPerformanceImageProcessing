cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/lib)

# Check if CUDA is enabled
option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    # Enable CUDA language
    enable_language(CUDA)
    
    # Find CUDA - try multiple methods for cluster compatibility
    find_package(CUDA QUIET)
    if(NOT CUDA_FOUND)
        # Try to find CUDA manually
        if(DEFINED ENV{CUDA_HOME})
            set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
        elseif(DEFINED ENV{CUDA_PATH})
            set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
        endif()
        
        # Enable CUDA language (this will find CUDA)
        enable_language(CUDA)
    endif()
    
    # Set CUDA properties
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set CUDA architecture (common GPU compute capabilities)
    # User can override with: cmake .. -DCMAKE_CUDA_ARCHITECTURES="75"
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80" CACHE STRING "CUDA architectures")
    endif()
    
    # CUDA source files
    set(CUDA_SOURCES
        src/cuda_kernels.cu
        src/cuda_utils.cpp
        src/filters_cuda.cpp
        src/edge_detection_cuda.cpp
    )
    
    # CUDA executable sources
    set(CUDA_EXE_SOURCES
        src/main_cuda.cpp
        src/image.cpp
        src/point_operations.cpp
        src/noise.cpp
        src/morphological.cpp
        src/geometric.cpp
        src/color_operations.cpp
    )
    
    # Create CUDA executable
    add_executable(image_processor_cuda ${CUDA_EXE_SOURCES} ${CUDA_SOURCES})
    
    # Set CUDA properties for the target
    set_property(TARGET image_processor_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    # Link CUDA libraries
    find_library(CUDART_LIBRARY 
        NAMES cudart
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
              ${CUDA_TOOLKIT_ROOT_DIR}/lib
              /usr/local/cuda/lib64
              /usr/local/cuda/lib
    )
    
    target_link_libraries(image_processor_cuda PRIVATE 
        ${CUDA_LIBRARIES}
        ${CUDART_LIBRARY}
    )
    
    # Add CUDA include directories
    target_include_directories(image_processor_cuda PRIVATE
        ${CUDA_TOOLKIT_ROOT_DIR}/include
    )
    
    # Platform-specific settings
    if(WIN32)
        target_compile_definitions(image_processor_cuda PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    
    message(STATUS "CUDA enabled - building image_processor_cuda")
    message(STATUS "CUDA Version: ${CUDA_VERSION}")
    message(STATUS "CUDA Toolkit Root Dir: ${CUDA_TOOLKIT_ROOT_DIR}")
    
else()
    # Regular CPU source files
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
    
    # Optional: Add OpenMP support
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(image_processor PUBLIC OpenMP::OpenMP_CXX)
    endif()
    
    message(STATUS "CUDA disabled - building regular image_processor")
endif()
