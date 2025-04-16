cmake_minimum_required(VERSION 3.18) # Use a more modern version for better features
project("yolo_kotlin_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set common CXX flags (apply to all targets unless overridden)
# Revert to default visibility, rely on explicit attributes in stub if needed
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fno-rtti -fvisibility=default")

# Enable Vulkan support definition for ncnn
add_definitions(-DNCNN_VULKAN=1)

# Enable OpenMP
find_package(OpenMP REQUIRED)
# Add OpenMP flags (needed for both compile and link)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include # Correctly points to app/src/main/cpp/ncnn_include
)

# Build the glslang stub as a STATIC library
add_library(glslang_stub STATIC
    glslang_stub.cpp
    glslang_stub.h # Include header for dependency tracking
)
# No need for target_compile_options if flags are set globally

# Find the Vulkan library
find_library(vulkan-lib vulkan HINTS ${ANDROID_NDK}/sources/third_party/vulkan/src/build/cmake/build/libs/${ANDROID_ABI})
if(NOT vulkan-lib)
    # Fallback to system library path if not found in NDK structure (less common now)
    find_library(vulkan-lib vulkan)
endif()
if(NOT vulkan-lib)
    message(FATAL_ERROR "Vulkan library not found. Check NDK setup and ensure target API level >= 24.")
else()
    message(STATUS "Found Vulkan library: ${vulkan-lib}")
endif()

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Import NCNN library (ensure path is correct)
add_library(ncnn STATIC IMPORTED GLOBAL)
# This path correctly resolves based on CMAKE_CURRENT_SOURCE_DIR and ANDROID_ABI
set(NCNN_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a")
if(NOT EXISTS ${NCNN_LIB_PATH})
    message(FATAL_ERROR "NCNN library not found at ${NCNN_LIB_PATH}")
endif()
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_LIB_PATH})
message(STATUS "Importing NCNN library from: ${NCNN_LIB_PATH}")

# Build the main native library
add_library(native-lib SHARED
    native-lib.cpp
)

# JNI functions in native-lib.cpp should use JNIEXPORT which handles visibility

# Link libraries to native-lib
target_link_libraries(native-lib
    PRIVATE # Use PRIVATE as these are implementation details
    # Link NCNN *before* the stub
    ncnn

    # Link glslang_stub static library using whole-archive to ensure symbols are pulled
    # For lld (default in modern NDKs):
    -Wl,--whole-archive
    glslang_stub
    -Wl,--no-whole-archive

    # Link OpenMP
    OpenMP::OpenMP_CXX

    # Link system libraries
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
    ${vulkan-lib}

    # Link atomic and math libraries (often needed by NCNN or dependencies)
    atomic
    m
)

