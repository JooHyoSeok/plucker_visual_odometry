cmake_minimum_required(VERSION 3.16)
project(vo_plucker_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video calib3d videoio imgcodecs features2d)

# g2o (third_party/g2o, git submodule, tag 20241228_git): BSD parts only = Eigen/dense solvers, no CHOLMOD/CSparse
# (LGPL), no OpenGL viewer (GPL), static, no apps/examples
foreach(opt G2O_USE_CHOLMOD G2O_USE_CSPARSE G2O_USE_LGPL_LIBS G2O_USE_OPENGL G2O_USE_LOGGING G2O_USE_OPENMP
            G2O_BUILD_APPS G2O_BUILD_EXAMPLES G2O_BUILD_SLAM2D_TYPES G2O_BUILD_SLAM2D_ADDON_TYPES G2O_BUILD_DATA_TYPES
            G2O_BUILD_SCLAM2D_TYPES G2O_BUILD_ICP_TYPES G2O_BUILD_SLAM3D_ADDON_TYPES BUILD_UNITTESTS G2O_BUILD_BENCHMARKS)
  set(${opt} OFF CACHE BOOL "" FORCE)
endforeach()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/g2o EXCLUDE_FROM_ALL)

add_library(vo_plucker src/geometry.cpp src/tracker.cpp src/frontend.cpp src/optimizer.cpp src/loop_closing.cpp
            src/local_mapping.cpp)
target_include_directories(vo_plucker PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vo_plucker PUBLIC Eigen3::Eigen ${OpenCV_LIBS} core stuff solver_eigen solver_dense types_sba types_slam3d
                      types_sim3)
target_compile_options(vo_plucker PRIVATE -Wall -Wextra)

add_executable(run_vo apps/run_vo.cpp)
target_link_libraries(run_vo vo_plucker)
target_compile_definitions(run_vo PRIVATE VP_LOOP_SERVER="${CMAKE_CURRENT_SOURCE_DIR}/../python/loop_server.py")

# viewer (optional): cmake ... -DPangolin_DIR=<Pangolin build or install dir>
find_package(Pangolin QUIET)
if(Pangolin_FOUND)
  add_library(vo_plucker_viewer src/viewer.cpp)
  target_link_libraries(vo_plucker_viewer PUBLIC vo_plucker ${Pangolin_LIBRARIES})
  target_compile_definitions(run_vo PRIVATE VP_WITH_VIEWER)
  target_link_libraries(run_vo vo_plucker_viewer)
else()
  message(STATUS "Pangolin not found: run_vo without viewer (set -DPangolin_DIR=...)")
endif()

add_executable(test_tracker tests/test_tracker.cpp)
target_link_libraries(test_tracker vo_plucker)
add_executable(test_frontend tests/test_frontend.cpp)
target_link_libraries(test_frontend vo_plucker)
add_executable(test_optimizer tests/test_optimizer.cpp)
target_link_libraries(test_optimizer vo_plucker)
