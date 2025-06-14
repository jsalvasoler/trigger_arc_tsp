cmake_minimum_required(VERSION 3.10)
project(TriggerArcTSP_CPP VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CMake policies
cmake_policy(SET CMP0167 NEW)  # Use new Boost package finding mechanism

# Build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Find required packages
find_package(Threads REQUIRED)
set(BOOST_ROOT "/opt/homebrew/opt/boost")
find_package(Boost REQUIRED)

# Add include directories
include_directories(include ${Boost_INCLUDE_DIRS})

# Add Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# Add source files
set(SOURCES
    src/instance.cpp
)

# Add executable
add_executable(main src/main.cpp ${SOURCES})
target_link_libraries(main ${Boost_LIBRARIES})

# Add test executable
add_executable(instance_tests tests/test_instance.cpp ${SOURCES})
target_link_libraries(instance_tests GTest::gtest_main ${Boost_LIBRARIES})

# Add test to CTest
add_test(NAME instance_tests COMMAND instance_tests)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Enable testing
enable_testing()

# Print configuration summary
message(STATUS "Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Boost version: ${Boost_VERSION}")
message(STATUS "  Output directories:")
message(STATUS "    - Binaries: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "    - Libraries: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "    - Archives: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}") 