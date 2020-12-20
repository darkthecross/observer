cmake_minimum_required(VERSION 3.10)

# set the project name
project(Imgui)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

###########################################
# IMGUI libraries
###########################################

add_library(imgui imgui.h imconfig.h imgui_internal.h imstb_textedit.h imgui.cpp)

add_library(imgui_impl_glfw imgui_impl_glfw.h imgui_impl_glfw.cpp)
target_link_libraries (imgui_impl_glfw imgui)
