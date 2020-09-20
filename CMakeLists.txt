cmake_minimum_required(VERSION 3.10)

# set the project name
project(Observer)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

###########################################
# Find packages.
###########################################

find_package(PCL 1.3 REQUIRED COMPONENTS common io filters visualization)

find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)

find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

###########################################
# Libraries
###########################################

add_library(util util.h util.cc)

###########################################
# Binaries
###########################################

add_executable(observer main.cc)
target_link_libraries(observer PUBLIC util realsense2 ${PCL_LIBRARIES})

add_executable(visualizer visualize.cc)
target_link_libraries(visualizer PUBLIC ${PCL_LIBRARIES})

add_executable(recorder recorder.cc)
target_link_libraries(recorder PUBLIC realsense2)

add_executable(replayer replayer.cc)
target_link_libraries(replayer PUBLIC util realsense2 ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})