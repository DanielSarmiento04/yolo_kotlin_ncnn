
            message(WARNING "Using `glslangTargets.cmake` is deprecated: use `find_package(glslang)` to find glslang CMake targets.")

            if (NOT TARGET glslang::glslang)
                include("/home/runner/work/ncnn/ncnn/build-riscv64/install/lib/cmake/glslang/glslang-targets.cmake")
            endif()

            if(OFF)
                add_library(glslang ALIAS glslang::glslang)
            else()
                add_library(glslang ALIAS glslang::glslang)
                add_library(MachineIndependent ALIAS glslang::MachineIndependent)
                add_library(GenericCodeGen ALIAS glslang::GenericCodeGen)
            endif()
        