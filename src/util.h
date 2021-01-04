#ifndef UTIL_H
#define UTIL_H

#define GLEW_STATIC

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

namespace util {

void InitOpenGL(int w, int h);

// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint for use
GLuint MatToTexture(cv::Mat mat, GLenum minFilter, GLenum magFilter, GLenum wrapFilter);

void DrawFrame(cv::Mat frame, size_t window_width, size_t window_height);

int64 GetTimestampMicros();

int64 GetTimestampNanos();

std::pair<cv::Mat, cv::Mat> ConvertFramesToMats(const std::pair<rs2::depth_frame, rs2::video_frame>& frameset);

}  // namespace util

#endif