cmake_minimum_required(VERSION 3.10.2)
project("MyARApp")

set(CMAKE_CXX_STANDARD 11)

# Disable Vulkan completely for NCNN at compile time
add_definitions(-DNCNN_VULKAN=0)

# Find and enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/ncnn)

# Compile the glslang stub separately with all symbols exported
add_library(glslang_stub STATIC
    glslang_stub.cpp
)

# Set visibility for the stub library
set_target_properties(glslang_stub PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

# This creates the shared library "native-lib" from native-lib.cpp
add_library(native-lib SHARED
    native-lib.cpp
)

# Define OpenMP library path - Android NDK r21+ includes libomp
set(OPENMP_LIB ${CMAKE_SOURCE_DIR}/libs/libomp.so)
if(NOT EXISTS ${OPENMP_LIB})
    # If file doesn't exist, use the NDK-provided one (in newer NDK versions)
    message(STATUS "Using NDK-provided OpenMP library")
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
endif()

# Only include NCNN library
add_library(ncnn_static STATIC IMPORTED)
set_target_properties(ncnn_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libncnn.a)

# Link with ncnn and our glslang stub
target_link_libraries(native-lib
    glslang_stub  # Link our stub first so its symbols are used
    ncnn_static
    ${OPENMP_LIB}
    jnigraphics
    android
    log
)

# Use additional link flags to handle symbol resolution issues
set_target_properties(native-lib PROPERTIES 
    LINK_FLAGS "-Wl,--allow-shlib-undefined -Wl,--allow-multiple-definition")

