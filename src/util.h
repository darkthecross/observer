#ifndef UTIL_H
#define UTIL_H

#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

namespace graphics_util {

void InitOpenGL(int w, int h);

// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint for use
GLuint MatToTexture(cv::Mat mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter);

void DrawFrame(cv::Mat frame, size_t window_width, size_t window_height);

}  // namespace graphics_util

#endif