#include "util.h"

#include <chrono>
#include <iostream>

namespace util {

using namespace std::chrono;

void InitOpenGL(int w, int h) {
  glViewport(0, 0, w, h);  // use a screen size of WIDTH x HEIGHT

  glMatrixMode(
      GL_PROJECTION);  // Make a simple 2D projection on the entire window
  glLoadIdentity();
  glOrtho(0.0, w, h, 0.0, 0.0, 100.0);

  glMatrixMode(GL_MODELVIEW);  // Set the matrix mode to object modeling

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClearDepth(0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Clear the window
}

// Function turn a cv::Mat into a texture, and return the texture ID as a GLuint
// for use
GLuint MatToTexture(cv::Mat mat, GLenum minFilter, GLenum magFilter,
                    GLenum wrapFilter) {
  // Generate a number for our textureID's unique handle
  GLuint textureID;
  glGenTextures(1, &textureID);

  // Bind to our texture handle
  glBindTexture(GL_TEXTURE_2D, textureID);

  // Catch silly-mistake texture interpolation method for magnification
  if (magFilter == GL_LINEAR_MIPMAP_LINEAR ||
      magFilter == GL_LINEAR_MIPMAP_NEAREST ||
      magFilter == GL_NEAREST_MIPMAP_LINEAR ||
      magFilter == GL_NEAREST_MIPMAP_NEAREST) {
    std::cout << "You can't use MIPMAPs for magnification - setting filter to "
                 "GL_LINEAR"
              << std::endl;
    magFilter = GL_LINEAR;
  }

  // Set texture interpolation methods for minification and magnification
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);

  // Set texture clamping method
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapFilter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapFilter);

  // Set incoming texture format to:
  // GL_BGR       for CV_CAP_OPENNI_BGR_IMAGE,
  // GL_LUMINANCE for CV_CAP_OPENNI_DISPARITY_MAP,
  // Work out other mappings as required ( there's a list in comments in main()
  // )
  GLenum inputColourFormat = GL_BGR;
  if (mat.channels() == 1) {
    inputColourFormat = GL_LUMINANCE;
  }

  // Create the texture
  glTexImage2D(GL_TEXTURE_2D,  // Type of texture
               0,       // Pyramid level (for mip-mapping) - 0 is the top level
               GL_RGB,  // Internal colour format to convert to
               mat.cols,  // Image width  i.e. 640 for Kinect in standard mode
               mat.rows,  // Image height i.e. 480 for Kinect in standard mode
               0,         // Border width in pixels (can either be 1 or 0)
               inputColourFormat,  // Input image format (i.e. GL_RGB, GL_RGBA,
                                   // GL_BGR etc.)
               GL_UNSIGNED_BYTE,   // Image data type
               mat.ptr());         // The actual image data itself

  // If we're using mipmaps then generate them. Note: This requires OpenGL 3.0
  // or higher
  if (minFilter == GL_LINEAR_MIPMAP_LINEAR ||
      minFilter == GL_LINEAR_MIPMAP_NEAREST ||
      minFilter == GL_NEAREST_MIPMAP_LINEAR ||
      minFilter == GL_NEAREST_MIPMAP_NEAREST) {
    glGenerateMipmap(GL_TEXTURE_2D);
  }

  return textureID;
}

void DrawFrame(cv::Mat frame, size_t window_width, size_t window_height) {
  // Clear color and depth buffers
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glMatrixMode(GL_MODELVIEW);  // Operate on model-view matrix

  glEnable(GL_TEXTURE_2D);
  GLuint image_tex =
      MatToTexture(frame, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_CLAMP);

  /* Draw a quad */
  glBegin(GL_QUADS);
  glTexCoord2i(0, 0);
  glVertex2i(0, 0);
  glTexCoord2i(0, 1);
  glVertex2i(0, window_height);
  glTexCoord2i(1, 1);
  glVertex2i(window_width, window_height);
  glTexCoord2i(1, 0);
  glVertex2i(window_width, 0);
  glEnd();

  glDeleteTextures(1, &image_tex);
  glDisable(GL_TEXTURE_2D);
}

int64 GetTimestampMicros() {
  return duration_cast<microseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

int64 GetTimestampNanos() {
  return duration_cast<nanoseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

}  // namespace util