cmake_minimum_required(VERSION 3.10)

# set the project name
project(Recorder LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(recorder recorder.cc)
target_link_libraries(recorder realsense2)
