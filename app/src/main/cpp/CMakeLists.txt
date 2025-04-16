cmake_minimum_required(VERSION 3.18)
project("yolo_kotlin_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for optimized builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fvisibility=hidden -O3 -ffast-math")

# Disable RTTI to reduce binary size (NCNN is built without RTTI)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

# Enable Vulkan support for NCNN
add_definitions(-DNCNN_VULKAN=1)

# Enable OpenMP for multi-threading
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")

message(STATUS "Using OpenMP flags: ${OpenMP_CXX_FLAGS}")

# Set proper include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include
)

# Build the glslang stub as a STATIC library
add_library(glslang_stub STATIC
    glslang_stub.cpp
    glslang_stub.h
)

target_compile_options(glslang_stub PRIVATE
    -fvisibility=default  # Override visibility for stub
)

# Find the Vulkan library
find_library(vulkan-lib vulkan)
if(NOT vulkan-lib)
    message(FATAL_ERROR "Vulkan library not found. Check NDK setup and ensure target API level >= 24.")
endif()
message(STATUS "Found Vulkan library: ${vulkan-lib}")

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Import NCNN library
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES 
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a
)

# Ensure the NCNN library exists at the specified path
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a)
    message(FATAL_ERROR "NCNN library not found at ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a")
endif()
message(STATUS "Using NCNN library: ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a")

# Build the main native library with JNI methods
add_library(native-lib SHARED
    native-lib.cpp
)

# Enable proper JNI symbol visibility
target_compile_options(native-lib PRIVATE
    -fvisibility=hidden  # Hide all symbols by default
)

# Define proper linking order
target_link_libraries(native-lib
    # Link NCNN first. It depends on symbols from glslang_stub.
    # Using --whole-archive for ncnn might still be needed if it has complex internal dependencies
    # or if symbols aren't being exported correctly otherwise, but let's keep it for now.
    -Wl,--whole-archive
    ncnn
    -Wl,--no-whole-archive

    # Link glslang stub AFTER ncnn. The linker should resolve symbols needed by ncnn from glslang_stub.
    # Remove --whole-archive for the stub; it's often not needed for simple stubs.
    glslang_stub

    # Link OpenMP
    OpenMP::OpenMP_CXX

    # Link Vulkan
    ${vulkan-lib}

    # Link system libraries
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}

    # Additional libraries that might be required for NCNN
    atomic
    m
)

