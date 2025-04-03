cmake_minimum_required(VERSION 3.18)
project(MyCudaProject LANGUAGES CXX CUDA)

# -------------------------
#  Compiler and Standard
# -------------------------

# If no build type is specified, default to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING 
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

add_executable(Recompression ${PROJECT_SOURCES})
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)
target_link_libraries(Recompression argparse)

# Optionally set properties on the target.
set_target_properties(Recompression PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES native
)