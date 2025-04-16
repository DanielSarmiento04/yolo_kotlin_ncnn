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

# Import Precompiled NCNN and related libraries
set(NCNN_PREBUILT_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI})

# Import NCNN library
add_library(ncnn STATIC IMPORTED)
set(NCNN_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libncnn.a)
if(NOT EXISTS ${NCNN_LIB_PATH})
    message(FATAL_ERROR "NCNN library not found at ${NCNN_LIB_PATH}")
endif()
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_LIB_PATH})
message(STATUS "Using NCNN library: ${NCNN_LIB_PATH}")

# Import glslang library
add_library(glslang STATIC IMPORTED)
set(GLSLANG_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libglslang.a)
if(NOT EXISTS ${GLSLANG_LIB_PATH})
    message(FATAL_ERROR "glslang library not found at ${GLSLANG_LIB_PATH}")
endif()
set_target_properties(glslang PROPERTIES IMPORTED_LOCATION ${GLSLANG_LIB_PATH})
message(STATUS "Using glslang library: ${GLSLANG_LIB_PATH}")

# Import SPIRV library
add_library(SPIRV STATIC IMPORTED)
set(SPIRV_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libSPIRV.a)
if(NOT EXISTS ${SPIRV_LIB_PATH})
    message(FATAL_ERROR "SPIRV library not found at ${SPIRV_LIB_PATH}")
endif()
set_target_properties(SPIRV PROPERTIES IMPORTED_LOCATION ${SPIRV_LIB_PATH})
message(STATUS "Using SPIRV library: ${SPIRV_LIB_PATH}")

# Import OGLCompiler library (Often needed by glslang)
add_library(OGLCompiler STATIC IMPORTED)
set(OGLCOMPILER_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libOGLCompiler.a)
if(NOT EXISTS ${OGLCOMPILER_LIB_PATH})
    message(FATAL_ERROR "OGLCompiler library not found at ${OGLCOMPILER_LIB_PATH}")
endif()
set_target_properties(OGLCompiler PROPERTIES IMPORTED_LOCATION ${OGLCOMPILER_LIB_PATH})
message(STATUS "Using OGLCompiler library: ${OGLCOMPILER_LIB_PATH}")

# Import OSDependent library (Often needed by glslang)
add_library(OSDependent STATIC IMPORTED)
set(OSDEPENDENT_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libOSDependent.a)
if(NOT EXISTS ${OSDEPENDENT_LIB_PATH})
    message(FATAL_ERROR "OSDependent library not found at ${OSDEPENDENT_LIB_PATH}")
endif()
set_target_properties(OSDependent PROPERTIES IMPORTED_LOCATION ${OSDEPENDENT_LIB_PATH})
message(STATUS "Using OSDependent library: ${OSDEPENDENT_LIB_PATH}")

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
    # Link NCNN first
    ncnn

    # Link glslang and its dependencies AFTER ncnn
    glslang
    SPIRV
    OGLCompiler
    OSDependent

    # Link OpenMP
    OpenMP::OpenMP_CXX

    # Link Vulkan
    ${vulkan-lib}

    # Link system libraries
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}

    # Additional libraries that might be required
    atomic
    m
)

