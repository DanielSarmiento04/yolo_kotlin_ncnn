cmake_minimum_required(VERSION 3.10.2)
project("MyARApp")

set(CMAKE_CXX_STANDARD 11)

# Disable Vulkan completely for NCNN
add_definitions(-DNCNN_VULKAN=OFF)

# Find and enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add OpenMP library directory
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/ncnn)

# This creates the shared library "native-lib" from native-lib.cpp.
add_library(native-lib SHARED
    native-lib.cpp
    glslang_stub.cpp  # Add our glslang stub implementation
)

# Define OpenMP library path - Android NDK r21+ includes libomp
set(OPENMP_LIB ${CMAKE_SOURCE_DIR}/libs/libomp.so)
if(NOT EXISTS ${OPENMP_LIB})
    # If file doesn't exist, use the NDK-provided one (in newer NDK versions)
    message(STATUS "Using NDK-provided OpenMP library")
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
endif()

# Only include NCNN library and remove dependency on glslang libraries
add_library(ncnn_static STATIC IMPORTED)
set_target_properties(ncnn_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libncnn.a)

# Link with simplified dependencies - remove Vulkan and glslang dependencies
target_link_libraries(native-lib
    ncnn_static
    ${OPENMP_LIB}
    jnigraphics
    android
    log
)

# Disable undefined symbol errors
set_target_properties(native-lib PROPERTIES LINK_FLAGS "-Wl,--allow-shlib-undefined")

