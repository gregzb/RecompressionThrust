cmake_minimum_required(VERSION 3.25.2)
project(MyCudaProject LANGUAGES CXX CUDA)

# -------------------------
#  Compiler and Standard
# -------------------------

# Require C++17 for the host compiler.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard (CUDA 12.x fully supports C++17 on device).
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
#  CUDA Architecture
# -------------------------
# For an RTX 2070 Super (compute capability 7.5).
set(CMAKE_CUDA_ARCHITECTURES 75)

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
add_executable(MyCudaProjectExe ${PROJECT_SOURCES})

# Optionally set properties on the target.
set_target_properties(MyCudaProjectExe PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "75"
)

# -------------------------
#  Data Handling (Optional)
# -------------------------
# If you want to install or copy the contents of "data/" somewhere, 
# you can add install or custom commands. For instance:
#
# install(DIRECTORY ${CMAKE_SOURCE_DIR}/data/
#         DESTINATION ${CMAKE_INSTALL_PREFIX}/data)

# -------------------------
#  Helpful Messages
# -------------------------
message(STATUS "Configuration complete. See usage instructions below.")