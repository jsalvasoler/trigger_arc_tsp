cmake_minimum_required(VERSION 3.10...3.27)
project(TriggerArcTSP_CPP VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CMake policies
# cmake_policy(SET CMP0167 NEW)  # Use new Boost package finding mechanism

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
find_package(Boost REQUIRED COMPONENTS program_options)

# Add include directories
include_directories(include ${Boost_INCLUDE_DIRS})

# Gurobi paths
if(APPLE)
    set(GUROBI_ROOT "/Library/gurobi1202/macos_universal2")
else()
    set(GUROBI_ROOT "/opt/gurobi12.0.2/linux64/")
endif()
set(GUROBI_LIB_DIR "${GUROBI_ROOT}/lib")
set(GUROBI_INCLUDE_DIR "${GUROBI_ROOT}/include")

# Include Gurobi headers
include_directories(${GUROBI_INCLUDE_DIR})

# Link Gurobi libraries
set(GUROBI_LIBRARIES
    ${GUROBI_LIB_DIR}/libgurobi_c++.a
    ${GUROBI_LIB_DIR}/libgurobi120.so  # Use .dylib instead on macOS
)

if(APPLE)
    set(GUROBI_LIBRARIES
        ${GUROBI_LIB_DIR}/libgurobi_c++.a
        ${GUROBI_LIB_DIR}/libgurobi120.dylib
    )
endif()

# Add Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

# Add source files
set(SOURCES
    src/instance.cpp
    src/tsp_model.cpp
    src/model.cpp
    src/method.cpp
    src/randomized_greedy.cpp
    src/mip_randomized_construction.cpp
)

# Add main executable
add_executable(trigger_arc_tsp src/main.cpp ${SOURCES})
target_link_libraries(trigger_arc_tsp ${Boost_LIBRARIES} ${GUROBI_LIBRARIES})

# Add test executables
add_executable(instance_tests tests/test_instance.cpp ${SOURCES})
target_link_libraries(instance_tests GTest::gtest_main ${Boost_LIBRARIES} ${GUROBI_LIBRARIES})

add_executable(tsp_model_tests tests/test_tsp_model.cpp ${SOURCES})
target_link_libraries(tsp_model_tests GTest::gtest_main ${Boost_LIBRARIES} ${GUROBI_LIBRARIES})

add_executable(model_tests tests/test_model.cpp ${SOURCES})
target_link_libraries(model_tests GTest::gtest_main ${Boost_LIBRARIES} ${GUROBI_LIBRARIES})

add_executable(randomized_greedy_tests tests/test_randomized_greedy.cpp ${SOURCES})
target_link_libraries(randomized_greedy_tests GTest::gtest_main ${Boost_LIBRARIES} ${GUROBI_LIBRARIES})

add_executable(mip_randomized_construction_tests tests/test_mip_randomized_construction.cpp ${SOURCES})
target_link_libraries(mip_randomized_construction_tests GTest::gtest_main ${Boost_LIBRARIES} ${GUROBI_LIBRARIES})

# Add tests to CTest
add_test(NAME instance_tests COMMAND instance_tests)
add_test(NAME tsp_model_tests COMMAND tsp_model_tests)
add_test(NAME model_tests COMMAND model_tests)
add_test(NAME randomized_greedy_tests COMMAND randomized_greedy_tests)
add_test(NAME mip_randomized_construction_tests COMMAND mip_randomized_construction_tests)

# Set env variables for Gurobi

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