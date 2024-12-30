cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Set C++ standard to 17 for Arrow compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)
find_package(Threads REQUIRED)

# Add DataToCpp include directory
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/cpp/DataToCpp
    ${Arrow_INCLUDE_DIRS}
    ${PARQUET_INCLUDE_DIRS}
)

# Add DataToCpp submodule
add_subdirectory(examples/cpp/DataToCpp)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    # Remove existing C++ standard flags from compiler options
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        SET(CMAKE_CXX_FLAGS "-Ofast -DHAVE_CXX0X -openmp -fpic -ftree-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        SET(CMAKE_CXX_FLAGS "-Ofast -lrt -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        SET(CMAKE_CXX_FLAGS "/O2 -DHAVE_CXX0X /W1 /openmp /EHsc")
    endif()

    # Create examples/cpp directory structure
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/examples/cpp)
    
    add_executable(example_search examples/cpp/example_search.cpp)
    target_link_libraries(example_search hnswlib)

    add_executable(example_build_and_save examples/cpp/example_build_and_save.cpp)
    target_link_libraries(example_build_and_save PRIVATE
        hnswlib
        arrow
        parquet
        Threads::Threads
        data2cpp
    )

    add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    target_link_libraries(example_epsilon_search hnswlib)

    add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    target_link_libraries(example_multivector_search hnswlib)

    add_executable(example_filter examples/cpp/example_filter.cpp)
    target_link_libraries(example_filter hnswlib)

    add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    target_link_libraries(example_replace_deleted hnswlib)

    add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    target_link_libraries(example_mt_search hnswlib)

    add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    target_link_libraries(example_mt_filter hnswlib)

    add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    target_link_libraries(example_mt_replace_deleted hnswlib)

    add_executable(example_load_and_search examples/cpp/example_load_and_search.cpp)
    target_link_libraries(example_load_and_search PRIVATE data2cpp hnswlib)

    # tests
    add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    target_link_libraries(multivector_search_test hnswlib)

    add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    target_link_libraries(epsilon_search_test hnswlib)

    add_executable(test_updates tests/cpp/updates_test.cpp)
    target_link_libraries(test_updates hnswlib)

    add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    target_link_libraries(searchKnnCloserFirst_test hnswlib)

    add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    target_link_libraries(searchKnnWithFilter_test hnswlib)

    add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    target_link_libraries(multiThreadLoad_test hnswlib)

    add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    target_link_libraries(multiThread_replace_test hnswlib)

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_link_libraries(main hnswlib)
endif()
