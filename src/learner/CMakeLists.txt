cmake_minimum_required(VERSION 3.10)

# set the project name
project(Replayer LANGUAGES CXX)

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

###########################################
# Binaries
###########################################

add_executable(image_extractor image_extractor.cc)
target_link_libraries(image_extractor realsense2 ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} glog::glog gflags)

add_executable(labeller labeller.cc)
target_link_libraries(labeller ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} glog::glog gflags)