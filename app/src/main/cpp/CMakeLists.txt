cmake_minimum_required(VERSION 3.18) # NDK 21+ recommended
project("yolo_kotlin_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags for optimized builds
# -fexceptions is needed by NCNN
# -fvisibility=hidden reduces binary size
# -O3 enables optimizations
# -ffast-math can improve performance but might affect precision slightly
# -fno-rtti is required as NCNN is built without RTTI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fvisibility=hidden -O3 -ffast-math -fno-rtti")

# Enable Vulkan support for NCNN
add_definitions(-DNCNN_VULKAN=1)

# Enable OpenMP for multi-threading
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "OpenMP found, enabling parallel processing.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # Note: Linker flags are typically handled by target_link_libraries with OpenMP::OpenMP_CXX
else()
    message(WARNING "OpenMP not found. Parallel processing will be disabled.")
endif()

# --- NCNN Configuration ---
# Set the path to the NCNN include directory relative to this CMakeLists.txt
set(NCNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include)
include_directories(${NCNN_INCLUDE_DIR})
message(STATUS "NCNN include directory: ${NCNN_INCLUDE_DIR}")

# Set the path to the prebuilt NCNN libraries for the target ABI
# ANDROID_ABI is automatically set by the Android Gradle plugin
set(NCNN_PREBUILT_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI})
message(STATUS "Looking for NCNN prebuilt libraries in: ${NCNN_PREBUILT_LIBS_DIR}")
if(NOT EXISTS ${NCNN_PREBUILT_LIBS_DIR})
    message(FATAL_ERROR "NCNN prebuilt libraries directory not found for ABI ${ANDROID_ABI} at ${NCNN_PREBUILT_LIBS_DIR}")
endif()

# Define NCNN library as IMPORTED STATIC
add_library(ncnn STATIC IMPORTED GLOBAL)
set(NCNN_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libncnn.a)
if(NOT EXISTS ${NCNN_LIB_PATH})
    message(FATAL_ERROR "libncnn.a not found at ${NCNN_LIB_PATH}")
endif()
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_LIB_PATH})
message(STATUS "Found NCNN library: ${NCNN_LIB_PATH}")

# Define glslang library as IMPORTED STATIC
add_library(glslang STATIC IMPORTED GLOBAL)
set(GLSLANG_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libglslang.a)
if(NOT EXISTS ${GLSLANG_LIB_PATH})
    message(FATAL_ERROR "libglslang.a not found at ${GLSLANG_LIB_PATH}")
endif()
set_target_properties(glslang PROPERTIES IMPORTED_LOCATION ${GLSLANG_LIB_PATH})
message(STATUS "Found glslang library: ${GLSLANG_LIB_PATH}")

# Define SPIRV library as IMPORTED STATIC
add_library(SPIRV STATIC IMPORTED GLOBAL)
set(SPIRV_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libSPIRV.a)
if(NOT EXISTS ${SPIRV_LIB_PATH})
    message(FATAL_ERROR "libSPIRV.a not found at ${SPIRV_LIB_PATH}")
endif()
set_target_properties(SPIRV PROPERTIES IMPORTED_LOCATION ${SPIRV_LIB_PATH})
message(STATUS "Found SPIRV library: ${SPIRV_LIB_PATH}")

# Define OGLCompiler library as IMPORTED STATIC
add_library(OGLCompiler STATIC IMPORTED GLOBAL)
set(OGLCOMPILER_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libOGLCompiler.a)
if(NOT EXISTS ${OGLCOMPILER_LIB_PATH})
    message(FATAL_ERROR "libOGLCompiler.a not found at ${OGLCOMPILER_LIB_PATH}")
endif()
set_target_properties(OGLCompiler PROPERTIES IMPORTED_LOCATION ${OGLCOMPILER_LIB_PATH})
message(STATUS "Found OGLCompiler library: ${OGLCOMPILER_LIB_PATH}")

# Define OSDependent library as IMPORTED STATIC
add_library(OSDependent STATIC IMPORTED GLOBAL)
set(OSDEPENDENT_LIB_PATH ${NCNN_PREBUILT_LIBS_DIR}/libOSDependent.a)
if(NOT EXISTS ${OSDEPENDENT_LIB_PATH})
    message(FATAL_ERROR "libOSDependent.a not found at ${OSDEPENDENT_LIB_PATH}")
endif()
set_target_properties(OSDependent PROPERTIES IMPORTED_LOCATION ${OSDEPENDENT_LIB_PATH})
message(STATUS "Found OSDependent library: ${OSDEPENDENT_LIB_PATH}")

# --- Android Libraries ---
# Find required Android system libraries
find_library(log-lib log) # For logging using __android_log_print
find_library(android-lib android) # For AAssetManager
find_library(jnigraphics-lib jnigraphics) # For interacting with Bitmaps if needed (not directly used here)

# Find the Vulkan library provided by the NDK
find_library(vulkan-lib vulkan)
if(NOT vulkan-lib)
    message(FATAL_ERROR "Vulkan library not found. Ensure NDK version supports Vulkan and target API level is >= 24.")
else()
    message(STATUS "Found Vulkan library: ${vulkan-lib}")
endif()

# --- Build Native Library ---
# Add your C++ source file(s)
add_library(native-lib SHARED
    native-lib.cpp
)

# Link libraries to your native-lib
target_link_libraries(native-lib
    # Link NCNN and its dependencies (order can matter)
    ncnn
    glslang
    SPIRV
    OGLCompiler
    OSDependent

    # Link OpenMP if found
    OpenMP::OpenMP_CXX

    # Link Vulkan library
    ${vulkan-lib}

    # Link Android system libraries
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}

    # Link standard C/C++ libraries if needed (often implicitly linked)
    # atomic might be needed by some NCNN operations or threading models
    atomic
    m # Math library
)

# Optional: Improve symbol visibility for JNI functions if needed,
# although -fvisibility=hidden in CXX_FLAGS usually requires explicit
# JNIEXPORT visibility which is handled by jni.h macros.
# target_compile_options(native-lib PRIVATE -fvisibility=default) # If symbols aren't found

