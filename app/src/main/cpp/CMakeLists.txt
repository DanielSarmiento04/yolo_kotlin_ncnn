cmake_minimum_required(VERSION 3.10.2)
project("yolo_kotlin_ncnn")

set(CMAKE_CXX_STANDARD 11)

# Enable Vulkan support
add_definitions(-DNCNN_VULKAN=1)

# Set proper include directories
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/ncnn)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/glslang)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/glslang/Include)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/glslang/SPIRV)

# Find and enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Compile the glslang stub with all symbols exported
add_library(glslang_stub STATIC
    glslang_stub.cpp
)

# Set visibility for the stub library
set_target_properties(glslang_stub PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

# Create the shared library "native-lib" from native-lib.cpp
add_library(native-lib SHARED
    native-lib.cpp
)

# Define OpenMP library path - Android NDK r21+ includes libomp
if(OpenMP_CXX_FOUND)
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
else()
    set(OPENMP_LIB ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libomp.so)
    if(NOT EXISTS ${OPENMP_LIB})
        message(FATAL_ERROR "OpenMP library not found. Please install NDK r21+ or provide libomp.so")
    endif()
endif()

# Include NCNN library with proper ABI path
add_library(ncnn_static STATIC IMPORTED)
set_target_properties(ncnn_static PROPERTIES IMPORTED_LOCATION 
    ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a)

# Include Vulkan library
find_library(vulkan-lib vulkan)
if(NOT vulkan-lib)
    message(FATAL_ERROR "Vulkan library not found")
endif()

# Link with ncnn, glslang stub, and other needed libraries
target_link_libraries(native-lib
    glslang_stub  # Link our stub first so its symbols are used
    ncnn_static
    ${OPENMP_LIB}
    ${vulkan-lib}
    jnigraphics
    android
    log
)

# Use additional link flags to handle symbol resolution issues
set_target_properties(native-lib PROPERTIES 
    LINK_FLAGS "-Wl,--allow-shlib-undefined -Wl,--allow-multiple-definition")

