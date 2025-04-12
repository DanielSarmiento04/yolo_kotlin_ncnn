cmake_minimum_required(VERSION 3.10.2) # Consider using a newer version like 3.18+ if available/needed
project("yolo_kotlin_ncnn")

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Enable C++ exceptions needed by NCNN and potentially the stub
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
# Disable RTTI if not needed (common for NCNN)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
# Set default visibility to public for exported symbols
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")


# Enable Vulkan support
add_definitions(-DNCNN_VULKAN=1)

# Enable OpenMP
find_package(OpenMP REQUIRED)
# Add OpenMP flags (needed for both compile and link)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# Note: target_link_libraries(OpenMP::OpenMP_CXX) handles linker flags later


# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ncnn_include
)

# Build the glslang stub as a STATIC library
add_library(glslang_stub STATIC
    glslang_stub.cpp
)
# No need for target_compile_options for visibility/rtti/exceptions if set globally via CMAKE_CXX_FLAGS

# Find the Vulkan library
find_library(vulkan-lib vulkan)
if(NOT vulkan-lib)
    message(FATAL_ERROR "Vulkan library not found. Check NDK setup and API level (min 24).")
endif()
message(STATUS "Found Vulkan library: ${vulkan-lib}")

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Import NCNN library (ensure path is correct)
add_library(ncnn STATIC IMPORTED GLOBAL)
set(NCNN_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/${ANDROID_ABI}/libncnn.a") # Use variable for clarity
if(NOT EXISTS ${NCNN_LIB_PATH})
    message(FATAL_ERROR "NCNN library not found at ${NCNN_LIB_PATH}")
endif()
set_target_properties(ncnn PROPERTIES IMPORTED_LOCATION ${NCNN_LIB_PATH})
message(STATUS "Importing NCNN library from: ${NCNN_LIB_PATH}")

# Build the main native library
add_library(native-lib SHARED
    native-lib.cpp
)
# No need for target_compile_options for visibility/rtti/exceptions if set globally

# Link libraries to native-lib
target_link_libraries(native-lib
    PRIVATE # Use PRIVATE unless these libs are needed by targets linking native-lib
    # Link NCNN *before* the stub
    ncnn

    # Link glslang_stub static library using whole-archive to ensure symbols are pulled
    # Note: CMake doesn't have direct whole-archive support, use linker flags
    # The exact flags might differ slightly based on the linker (lld vs ld)
    # For lld (default in modern NDKs):
    -Wl,--whole-archive # Start whole archive section
    glslang_stub        # Link the stub library
    -Wl,--no-whole-archive # End whole archive section

    # Link OpenMP
    OpenMP::OpenMP_CXX

    # Link system libraries
    ${log-lib}
    ${android-lib}
    ${jnigraphics-lib}
    ${vulkan-lib}

    # Link C++ Standard Library (shared version)
    c++_shared

    # Link atomic and math libraries (often needed by NCNN or dependencies)
    atomic
    m
)

