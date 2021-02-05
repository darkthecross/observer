cmake_minimum_required(VERSION 3.10)

# set the project name
project(Observer LANGUAGES CUDA CXX)

add_subdirectory(src/recorder)

add_subdirectory(src/replayer)

add_subdirectory(src/learner)