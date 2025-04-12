cmake_minimum_required(VERSION 3.10.2)
project("yolo_kotlin_ncnn")

set(CMAKE_CXX_STANDARD 11)

# Enable Vulkan for NCNN at compile time
add_definitions(-DNCNN_VULKAN=1)

# Set proper include directories for ncnn and glslang
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/ncnn)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/glslang)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/glslang/SPIRV)

# Add this after the initial include_directories block

# Create symbolic link to ensure NCNN finds our stub header
if(EXISTS ${CMAKE_SOURCE_DIR}/ncnn_include/glslang/SPIRV)
    # Only create if the directory exists but our file doesn't
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/ncnn_include/glslang/SPIRV/GlslangToSpv.h)
        # Copy our stub to the actual include location
        configure_file(${CMAKE_SOURCE_DIR}/glslang_stub.h 
                      ${CMAKE_SOURCE_DIR}/ncnn_include/glslang/SPIRV/GlslangToSpv.h 
                      COPYONLY)
        message(STATUS "Created glslang stub in include directory")
    endif()
endif()


# Find and enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Compile the glslang stub separately with all symbols exported
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
set(OPENMP_LIB ${CMAKE_SOURCE_DIR}/libs/libomp.so)
if(NOT EXISTS ${OPENMP_LIB})
    # If file doesn't exist, use the NDK-provided one (in newer NDK versions)
    message(STATUS "Using NDK-provided OpenMP library")
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
endif()

# Include NCNN library
add_library(ncnn_static STATIC IMPORTED)
set_target_properties(ncnn_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libncnn.a)

# Include Vulkan library
find_library(vulkan-lib vulkan)

# Link with ncnn, our glslang stub, and Vulkan
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

