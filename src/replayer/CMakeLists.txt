cmake_minimum_required(VERSION 3.10)

# set the project name
project(Replayer LANGUAGES CUDA CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

###########################################
# Find packages.
###########################################

find_package(OpenCV REQUIRED)

find_package(glog 0.4.0 REQUIRED)

find_package(gflags REQUIRED)

find_package(Boost COMPONENTS filesystem REQUIRED)

find_package(glfw3 3.2 REQUIRED)
find_package(OpenGL 3.0 REQUIRED)
find_package(GLEW REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

###########################################
# Libraries
###########################################

add_library(util STATIC util.h util.cc)
target_link_libraries(util glfw OpenGL::OpenGL ${OpenCV_LIBRARIES} ${GLEW_LIBRARIES} realsense2)

add_library(gpu STATIC gpu.h gpu.cu)
set_target_properties(gpu
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(frame_analyzer STATIC frame_analyzer.h frame_analyzer.cc)
target_link_libraries(frame_analyzer util ${OpenCV_LIBRARIES} realsense2 gpu)
set_target_properties(frame_analyzer
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

###########################################
# Binaries
###########################################

add_executable(replayer replayer.cc)
target_link_libraries(replayer realsense2 ${OpenCV_LIBRARIES})

add_executable(replayer_async replayer_async.cc)
target_link_libraries(replayer_async util frame_analyzer realsense2 glfw OpenGL::OpenGL ${OpenCV_LIBRARIES} ${GLEW_LIBRARIES} ${Boost_LIBRARIES} glog::glog gflags)
set_target_properties(replayer_async
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(frame_analyzer_dev frame_analyzer_dev.cc)
set_target_properties(frame_analyzer_dev
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(frame_analyzer_dev util frame_analyzer realsense2 ${OpenCV_LIBRARIES} glog::glog gflags)
