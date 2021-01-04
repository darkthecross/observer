#ifndef FRAME_ANALYZER_H
#define FRAME_ANALYZER_H

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace frame_analyzer {

// Max distance under consideration in meters.
constexpr float MAX_DISTANCE = 5.12;
constexpr size_t FRAME_BUFFER_SIZE = 5;
// constexpr float CONSIDERED_DISTANCE = 4.0;

struct AnalyzerOutput {
  cv::Mat marked_img;
  size_t frame_count;
};

class FrameAnalyzer {
 public:
  explicit FrameAnalyzer(float depth_scale = 1e-3);

  void AnalyzeDepthMat(cv::Mat depth_mat, AnalyzerOutput* output);
  void AnalyzeIRMat(cv::Mat ir_mat, AnalyzerOutput* output);

  AnalyzerOutput AnalyzeFrames(
      const std::pair<rs2::depth_frame, rs2::video_frame>& frameset);

  AnalyzerOutput AnalyzeMats(const std::pair<cv::Mat, cv::Mat>& matset);

  size_t GetNumFramesetsProcessed() { return frameset_count_; }

  // micro secs.
  float GetAverageFramesetAnalyzeTime() { return avg_frameset_analyze_time_; }

 private:
  std::pair<cv::Mat, cv::Mat> ConvertFramesToMats(
      const std::pair<rs2::depth_frame, rs2::video_frame>& frameset);

  size_t frameset_count_ = 0;
  // Depth scale in meters.
  float avg_frameset_analyze_time_ = 0.0;

  float depth_scale_ = 1e-3;
  std::vector<cv::Mat> original_depth_frame_buffer_;

  cv::Ptr<cv::SimpleBlobDetector> ir_blob_detector_;
  cv::Ptr<cv::SimpleBlobDetector> depth_blob_detector_;
};

}  // namespace frame_analyzer

#endif