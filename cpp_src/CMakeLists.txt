cmake_minimum_required(VERSION 3.10...3.27)
project(TriggerArcTSP_CPP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_GUROBI "Enable Gurobi solver" OFF)  # default OFF since you don't have Gurobi

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

find_package(Threads REQUIRED)
set(BOOST_ROOT "/opt/homebrew/opt/boost")
find_package(Boost REQUIRED COMPONENTS program_options)

include_directories(include ${Boost_INCLUDE_DIRS})

if(USE_GUROBI)
    message(STATUS "Gurobi support enabled")
    set(GUROBI_ROOT "/Library/gurobi1202/macos_universal2")
    set(GUROBI_LIB_DIR "${GUROBI_ROOT}/lib")
    set(GUROBI_INCLUDE_DIR "${GUROBI_ROOT}/include")
    include_directories(${GUROBI_INCLUDE_DIR})

    if(APPLE)
        set(GUROBI_LIBRARIES
            ${GUROBI_LIB_DIR}/libgurobi_c++.a
            ${GUROBI_LIB_DIR}/libgurobi120.dylib
        )
    else()
        set(GUROBI_LIBRARIES
            ${GUROBI_LIB_DIR}/libgurobi_c++.a
            ${GUROBI_LIB_DIR}/libgurobi120.so
        )
    endif()
else()
    message(STATUS "Gurobi support disabled")
endif()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

# Source files common to all builds
set(SOURCES_COMMON
    src/instance.cpp
    src/model.cpp
    src/method.cpp
    src/randomized_greedy.cpp
)

# Add tsp_model.cpp only if Gurobi enabled
if(USE_GUROBI)
    list(APPEND SOURCES_COMMON src/tsp_model.cpp)
endif()

add_executable(trigger_arc_tsp src/main.cpp ${SOURCES_COMMON})

target_link_libraries(trigger_arc_tsp PRIVATE ${Boost_LIBRARIES})

if(USE_GUROBI)
    target_link_libraries(trigger_arc_tsp PRIVATE ${GUROBI_LIBRARIES})
endif()

# Tests (same sources as main executable)
add_executable(instance_tests tests/test_instance.cpp ${SOURCES_COMMON})
target_link_libraries(instance_tests PRIVATE GTest::gtest_main ${Boost_LIBRARIES})
if(USE_GUROBI)
    target_link_libraries(instance_tests PRIVATE ${GUROBI_LIBRARIES})
endif()

add_executable(tsp_model_tests tests/test_tsp_model.cpp ${SOURCES_COMMON})
target_link_libraries(tsp_model_tests PRIVATE GTest::gtest_main ${Boost_LIBRARIES})
if(USE_GUROBI)
    target_link_libraries(tsp_model_tests PRIVATE ${GUROBI_LIBRARIES})
endif()

add_executable(model_tests tests/test_model.cpp ${SOURCES_COMMON})
target_link_libraries(model_tests PRIVATE GTest::gtest_main ${Boost_LIBRARIES})
if(USE_GUROBI)
    target_link_libraries(model_tests PRIVATE ${GUROBI_LIBRARIES})
endif()

add_executable(randomized_greedy_tests tests/test_randomized_greedy.cpp ${SOURCES_COMMON})
target_link_libraries(randomized_greedy_tests PRIVATE GTest::gtest_main ${Boost_LIBRARIES})
if(USE_GUROBI)
    target_link_libraries(randomized_greedy_tests PRIVATE ${GUROBI_LIBRARIES})
endif()

add_test(NAME instance_tests COMMAND instance_tests)
add_test(NAME tsp_model_tests COMMAND tsp_model_tests)
add_test(NAME model_tests COMMAND model_tests)
add_test(NAME randomized_greedy_tests COMMAND randomized_greedy_tests)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

enable_testing()

message(STATUS "Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Boost version: ${Boost_VERSION}")
message(STATUS "  Gurobi enabled: ${USE_GUROBI}")
message(STATUS "  Output directories:")
message(STATUS "    - Binaries: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "    - Libraries: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "    - Archives: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
