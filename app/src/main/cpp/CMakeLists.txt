cmake_minimum_required(VERSION 3.10.2)
project("MyARApp")

set(CMAKE_CXX_STANDARD 11)

# Find and enable OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add OpenMP library directory
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/glslang)
include_directories(${CMAKE_SOURCE_DIR}/ncnn_include/ncnn)

# This creates the shared library "native-lib" from native-lib.cpp.
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

# Ensure the correct order of glslang libraries (this order matters)
add_library(glslang_static STATIC IMPORTED)
set_target_properties(glslang_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libglslang.a)

add_library(spirv_static STATIC IMPORTED)
set_target_properties(spirv_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libSPIRV.a)

add_library(oglcompiler_static STATIC IMPORTED)
set_target_properties(oglcompiler_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libOGLCompiler.a)

add_library(ncnn_static STATIC IMPORTED)
set_target_properties(ncnn_static PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/libs/libncnn.a)

# Link against all libraries with correct dependency order
target_link_libraries(native-lib
    ncnn_static
    glslang_static
    spirv_static
    oglcompiler_static
    ${OPENMP_LIB}
    vulkan
    jnigraphics
    android
    log
)

# Disable undefined symbol errors if OpenMP still causes issues
set_target_properties(native-lib PROPERTIES LINK_FLAGS "-Wl,--allow-shlib-undefined")