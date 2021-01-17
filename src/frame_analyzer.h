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

struct FeaturePoint {
  float x;
  float y;
  float r;

  FeaturePoint(float xx, float yy, float rr) : x(xx), y(yy), r(rr) {}
};

struct FeatureObject {
  float position[3];
  float velocity[3];
  int unique_id;
  float belief;
};

struct IROutput {
  std::vector<FeaturePoint> feature_points;
};

struct DepthOutput {
  std::vector<FeaturePoint> filtered_feature_points;
};

struct AnalyzerOutput {
  cv::Mat marked_img;
  std::vector<FeatureObject> objects;
  size_t frame_count;
};

class FrameAnalyzer {
 public:
  explicit FrameAnalyzer(rs2_intrinsics intrinsics, float depth_scale = 1e-3);

  // Stage 1: Extract light blobs from ir_mat.
  IROutput AnalyzeIRMat(cv::Mat ir_mat);

  // Stage 2: Filter IROutput based on depth image.
  DepthOutput AnalyzeDepthMat(cv::Mat depth_mat, const IROutput &ir_output);

  // Stage 3: Aggregate to objects.
  AnalyzerOutput ProcessFeaturePoints(const DepthOutput& depth_output);

  AnalyzerOutput AnalyzeFrames(
      const std::pair<rs2::depth_frame, rs2::video_frame>& frameset);

  AnalyzerOutput CorePipeline(const std::pair<cv::Mat, cv::Mat>& matset);

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
  std::vector<std::vector<FeaturePoint>> feature_points_buffer_;

  cv::Ptr<cv::SimpleBlobDetector> ir_blob_detector_;

  rs2_intrinsics depth_intrinsics_;
};

}  // namespace frame_analyzer

#endif