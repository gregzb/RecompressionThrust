cmake_minimum_required(VERSION 3.25.2)
project(MyCudaProject LANGUAGES CXX CUDA)

# -------------------------
#  Compiler and Standard
# -------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If no build type is specified, default to Debug.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING 
        "Choose the build type (e.g. Debug, Release, RelWithDebInfo, MinSizeRel)" 
        FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# -------------------------
#  Build Flags
# -------------------------

# Host compiler flags
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -fno-omit-frame-pointer")

# NVCC flags
set(CMAKE_CUDA_FLAGS_DEBUG   "-G -O0 -g --extended-lambda")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --extended-lambda")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-G -O3 -g -fno-omit-frame-pointer --extended-lambda")

# -------------------------
#  Include Directories
# -------------------------
# Point to your include directory.
include_directories(${CMAKE_SOURCE_DIR}/include)

# -------------------------
#  Source Files
# -------------------------
# Recursively collect *.cpp and *.cu files from "src".
file(GLOB_RECURSE PROJECT_SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
)

# -------------------------
#  Executable
# -------------------------
# find_package(OpenMP REQUIRED)

add_executable(MyCudaProjectExe ${PROJECT_SOURCES})

# target_link_libraries(MyCudaProjectExe PRIVATE OpenMP::OpenMP_CXX)
# target_compile_options(MyCudaProjectExe PRIVATE 
#   $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler;-fopenmp>
# )

include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)
target_link_libraries(MyCudaProjectExe argparse)

# Optionally set properties on the target.
set_target_properties(MyCudaProjectExe PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES native
)