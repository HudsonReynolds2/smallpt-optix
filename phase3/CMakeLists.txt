cmake_minimum_required(VERSION 3.18)
project(optix_smallpt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.1.0" CACHE PATH "OptiX install root")
find_path(OPTIX_INCLUDE_DIR optix.h HINTS "${OPTIX_ROOT}/include" REQUIRED)
message(STATUS "OptiX include: ${OPTIX_INCLUDE_DIR}")

set(SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders.cu")
set(SHADER_PTX "${CMAKE_CURRENT_BINARY_DIR}/shaders.ptx")

add_custom_command(
    OUTPUT  ${SHADER_PTX}
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx -arch=sm_86 -O3 --use_fast_math -std=c++17
            -I "${OPTIX_INCLUDE_DIR}" -I "${CMAKE_CURRENT_SOURCE_DIR}/src"
            -o "${SHADER_PTX}" "${SHADER_SRC}"
    DEPENDS ${SHADER_SRC} "${CMAKE_CURRENT_SOURCE_DIR}/src/shared.h"
    COMMENT "Compiling shaders.cu to PTX"
)
add_custom_target(shaders ALL DEPENDS ${SHADER_PTX})

add_executable(optix_smallpt src/main.cpp)
add_dependencies(optix_smallpt shaders)
target_include_directories(optix_smallpt PRIVATE src "${OPTIX_INCLUDE_DIR}")
target_link_libraries(optix_smallpt PRIVATE CUDA::cudart)
target_compile_options(optix_smallpt PRIVATE /O2 /W3)
set_target_properties(optix_smallpt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
