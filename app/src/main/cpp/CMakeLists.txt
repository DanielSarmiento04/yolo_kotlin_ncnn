cmake_minimum_required(VERSION 3.10.2)
project("yolo_kotlin_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Vulkan support
add_definitions(-DNCNN_VULKAN=1)

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include/ncnn
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include/glslang
)

# Build the glslang stub library as STATIC
add_library(glslang_stub STATIC
    glslang_stub.cpp
)

# Set properties for glslang_stub (Visibility settings less critical for STATIC)
set_target_properties(glslang_stub PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Find the Vulkan library
find_library(vulkan-lib vulkan)

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Import NCNN library
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a)

# Build the main native library
add_library(native-lib SHARED
    native-lib.cpp
)

# Link against our libraries
# Link static libraries directly. Order might matter.
target_link_libraries(native-lib
    # System/Shared libs first
    ${vulkan-lib}
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
    OpenMP::OpenMP_CXX

    # Static libraries - glslang_stub needs to satisfy ncnn's dependencies
    glslang_stub
    ncnn
)

# Add specific link flags to handle symbol resolution
# Keep --no-undefined for strictness
set_target_properties(native-lib PROPERTIES
    LINK_FLAGS "-Wl,--allow-multiple-definition -Wl,-z,nodelete -Wl,--no-undefined")

