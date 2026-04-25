cmake_minimum_required(VERSION 3.18)
project(smallpt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# SM 8.6 = RTX 3080 Ti (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDAToolkit REQUIRED)

add_executable(smallpt src/kernel.cu)

target_include_directories(smallpt PRIVATE src)

target_link_libraries(smallpt PRIVATE CUDA::curand)

target_compile_options(smallpt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
    >
)

set_target_properties(smallpt PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
