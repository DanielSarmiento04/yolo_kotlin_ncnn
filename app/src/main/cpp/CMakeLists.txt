cmake_minimum_required(VERSION 3.10.2)
project("yolo_kotlin_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set global C++ flags BEFORE defining libraries
# -fvisibility=hidden: Make symbols hidden by default
# -fno-rtti -fno-exceptions: Match common NCNN build settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fno-rtti -fno-exceptions")

# Enable Vulkan support
add_definitions(-DNCNN_VULKAN=1)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include
)

# Build the glslang stub library as STATIC
add_library(glslang_stub STATIC
    glslang_stub.cpp
)
# Symbols within glslang_stub.cpp marked with __attribute__((visibility("default"))) will be exported

# Find the Vulkan library
find_library(vulkan-lib vulkan)
if(NOT vulkan-lib)
    message(FATAL_ERROR "Vulkan library not found. Check NDK setup and API level.")
endif()
message(STATUS "Found Vulkan library: ${vulkan-lib}")

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Import NCNN library
add_library(ncnn STATIC IMPORTED GLOBAL)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a)
message(STATUS "Importing NCNN library from: ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a")

# Build the main native library
add_library(native-lib SHARED
    native-lib.cpp
)

# Link libraries to native-lib
target_link_libraries(native-lib
    PRIVATE
    # System libraries
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
    ${vulkan-lib}

    # OpenMP
    OpenMP::OpenMP_CXX

    # Force inclusion of all objects from glslang_stub static library
    -Wl,--whole-archive
    glslang_stub
    -Wl,--no-whole-archive

    # Link NCNN *after* glslang_stub
    ncnn

    # Other dependencies
    atomic
    m
)

