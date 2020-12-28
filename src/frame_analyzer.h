#ifndef FRAME_ANALYZER_H
#define FRAME_ANALYZER_H

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

namespace frame_analyzer {

struct AnalyzerOutput {
  cv::Mat marked_img;
  size_t frame_count;
};

class DepthFrameAnalyzer {
 public:
  explicit DepthFrameAnalyzer(float depth_scale = 1e-3)
      : depth_scale_(depth_scale) {}

  AnalyzerOutput AnalyzeFrame(rs2::frame input_frame);

  size_t GetNumFramesProcessed() { return frame_count_; }

 private:
  size_t frame_count_ = 0;
  // Depth scale in meters.
  float depth_scale_ = 1e-3;
};

}  // namespace frame_analyzer

#endif