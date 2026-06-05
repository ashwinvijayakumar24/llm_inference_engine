cmake_minimum_required(VERSION 3.18)
project(engine_kernels CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module)

# nanobind — fetch if not present
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v2.1.0
)
FetchContent_MakeAvailable(nanobind)

# CUDA kernel object
add_library(add_one_kernel OBJECT add_one_kernel.cu)
set_target_properties(add_one_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(add_one_kernel PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Python extension
nanobind_add_module(engine_kernels
    bindings.cpp
    $<TARGET_OBJECTS:add_one_kernel>
)
target_link_libraries(engine_kernels PRIVATE CUDA::cudart)
target_include_directories(engine_kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
