# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(MLFramework)
set(CMAKE_CXX_STANDARD 17)

# Include directories (your headers)
include_directories(include)

# Add the source files
file(GLOB SOURCES "src/*.cpp")

# Create an executable
add_executable(ml_framework ${SOURCES})
