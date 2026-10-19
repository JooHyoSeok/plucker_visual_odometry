cmake_minimum_required(VERSION 3.16)
project(vo_plucker_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video calib3d videoio imgcodecs)

add_library(vo_plucker src/geometry.cpp src/tracker.cpp src/frontend.cpp)
target_include_directories(vo_plucker PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vo_plucker PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_compile_options(vo_plucker PRIVATE -Wall -Wextra)

add_executable(run_vo apps/run_vo.cpp)
target_link_libraries(run_vo vo_plucker)

add_executable(test_tracker tests/test_tracker.cpp)
target_link_libraries(test_tracker vo_plucker)
add_executable(test_frontend tests/test_frontend.cpp)
target_link_libraries(test_frontend vo_plucker)
