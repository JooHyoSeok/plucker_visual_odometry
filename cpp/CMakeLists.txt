cmake_minimum_required(VERSION 3.16)
project(vo_plucker_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(vo_plucker src/geometry.cpp src/tracker.cpp)
target_include_directories(vo_plucker PUBLIC include)
target_link_libraries(vo_plucker PUBLIC Eigen3::Eigen)
target_compile_options(vo_plucker PRIVATE -Wall -Wextra)

add_executable(test_tracker tests/test_tracker.cpp)
target_link_libraries(test_tracker vo_plucker)
