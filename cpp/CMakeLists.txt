# Cross-platform build for the nanochat-core C++ port.
# Works with MSVC (Visual Studio 2022), GCC and Clang on any OS.
# OpenMP is used automatically if available (optional).
#
#   cmake -S . -B build
#   cmake --build build --config Release
#
cmake_minimum_required(VERSION 3.15)
project(nanochat_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(OpenMP QUIET)

function(configure tgt)
    if(MSVC)
        target_compile_options(${tgt} PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(${tgt} PRIVATE -O3 -ffast-math)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

# float + OpenMP: train / sample driver
add_executable(nanochat main.cpp)
configure(nanochat)

# double precision: gradient check (backprop correctness)
add_executable(nanochat_gradcheck test_grad.cpp)
target_compile_definitions(nanochat_gradcheck PRIVATE GPT_USE_DOUBLE)
configure(nanochat_gradcheck)
