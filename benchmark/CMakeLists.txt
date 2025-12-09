cmake_minimum_required(VERSION 3.10)
project(hnswlib_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# Find OpenBLAS (required)
find_package(BLAS REQUIRED)
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")

# Include hnswlib headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../hnswlib)

# Benchmark executable
add_executable(benchmark_ip benchmark_ip.cpp)
target_link_libraries(benchmark_ip ${BLAS_LIBRARIES})
target_compile_definitions(benchmark_ip PRIVATE USE_BLAS)
