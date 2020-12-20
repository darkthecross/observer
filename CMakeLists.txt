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

INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

###########################################
# IMGUI libraries
###########################################

add_subdirectory(src/imgui/)

###########################################
# Binaries
###########################################

add_executable(recorder src/recorder.cc)
target_link_libraries(recorder PUBLIC realsense2)

add_executable(replayer src/replayer.cc)
target_link_libraries(replayer PUBLIC realsense2 ${OpenCV_LIBRARIES})

add_executable(replayer_async src/replayer_async.cc)
target_link_libraries(replayer_async PUBLIC realsense2 ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} glog::glog gflags imgui_impl_glfw)
