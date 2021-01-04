cmake_minimum_required(VERSION 3.10)

# set the project name
project(Observer)

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

find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL 3.0 REQUIRED)
find_package(GLEW REQUIRED)

###########################################
# Libraries
###########################################

add_library(util STATIC src/util.h src/util.cc)
target_link_libraries(util glfw OpenGL::OpenGL ${OpenCV_LIBRARIES} ${GLEW_LIBRARIES} realsense2)

add_library(frame_analyzer STATIC src/frame_analyzer.h src/frame_analyzer.cc)
target_link_libraries(frame_analyzer util ${OpenCV_LIBRARIES} realsense2)

###########################################
# Binaries
###########################################

add_executable(recorder src/recorder.cc)
target_link_libraries(recorder realsense2)

add_executable(replayer src/replayer.cc)
target_link_libraries(replayer realsense2 ${OpenCV_LIBRARIES})

add_executable(replayer_async src/replayer_async.cc)
target_link_libraries(replayer_async util frame_analyzer realsense2 glfw OpenGL::OpenGL ${OpenCV_LIBRARIES} ${GLEW_LIBRARIES} ${Boost_LIBRARIES} glog::glog gflags)

add_executable(frame_analyzer_dev src/frame_analyzer_dev.cc)
target_link_libraries(frame_analyzer_dev util frame_analyzer realsense2 ${OpenCV_LIBRARIES} glog::glog gflags)