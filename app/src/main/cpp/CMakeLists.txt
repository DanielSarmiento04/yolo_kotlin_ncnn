cmake_minimum_required(VERSION 3.10.2)
project("MyARApp")

set(CMAKE_CXX_STANDARD 11)

# This creates the shared library "native-lib" from native-lib.cpp.
add_library(native-lib SHARED
    native-lib.cpp
)

# Additional configuration like including NCNN headers:
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/glslang)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/ncnn)

# Define each prebuilt static library individually
add_library(ncnn_static STATIC IMPORTED)
set_target_properties(ncnn_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libncnn.a)

add_library(oglcompiler_static STATIC IMPORTED)
set_target_properties(oglcompiler_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libOGLCompiler.a)

add_library(spirv_static STATIC IMPORTED)
set_target_properties(spirv_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libSPIRV.a)

add_library(glslang_static STATIC IMPORTED)
set_target_properties(glslang_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libglslang.a)

# Link against all libraries
target_link_libraries(native-lib
    ncnn_static
    oglcompiler_static
    spirv_static
    glslang_static
    vulkan
    jnigraphics
    android
    log
)