cmake_minimum_required(VERSION 3.10.2)
project("yolo_kotlin_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Vulkan support
add_definitions(-DNCNN_VULKAN=1)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include
)

# Build the glslang stub back as a STATIC library
add_library(glslang_stub STATIC
    glslang_stub.cpp
)
# Ensure symbols are exported and apply consistent compile options
target_compile_options(glslang_stub PRIVATE -fvisibility=default -fno-rtti -fexceptions)

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

# Build the main native library (only native-lib.cpp source here)
add_library(native-lib SHARED
    native-lib.cpp
)

# Explicitly disable RTTI and enable exceptions for native-lib
target_compile_options(native-lib PRIVATE -fno-rtti -fexceptions)

# Link libraries to native-lib
target_link_libraries(native-lib
    PRIVATE
    # Link OpenMP first
    OpenMP::OpenMP_CXX

    # Link NCNN *before* the stub
    ncnn

    # Link glslang_stub static library with whole-archive
    -Wl,--whole-archive
    glslang_stub
    -Wl,--no-whole-archive

    # Link system libraries, C++ STL, and math libraries last
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
    ${vulkan-lib}
    c++_shared
    atomic
    m
)

