cmake_minimum_required(VERSION 3.20)
project(smallpt CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(smallpt smallpt.cpp)

# ----- Optimization flags -----
if(MSVC)
    target_compile_options(smallpt PRIVATE
        /O2          # Full optimization (equiv to -O2; use /Ox for max)
        /fp:fast     # Fast floating point
        /GL          # Whole-program optimization
        /arch:AVX2   # AVX2 on your 5900X
        /W3
    )
    target_link_options(smallpt PRIVATE /LTCG)
else()
    # GCC / Clang (e.g. via MinGW or clang-cl)
    target_compile_options(smallpt PRIVATE
        -O3
        -march=native
        -ffast-math
        -Wall
    )
endif()

# ----- OpenMP -----
find_package(OpenMP REQUIRED)
target_link_libraries(smallpt PRIVATE OpenMP::OpenMP_CXX)

# ----- _USE_MATH_DEFINES for M_PI on MSVC, suppress fopen warning -----
target_compile_definitions(smallpt PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)

message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
