#include "frame_analyzer.h"

#include <iostream>
#include <string>
#include <thread>

#include "util.h"

namespace frame_analyzer {

using namespace cv;

FrameAnalyzer::FrameAnalyzer(float depth_scale) {
  depth_scale_ = depth_scale;
  // For IR, we detect a light blob which is quite obvious.
  cv::SimpleBlobDetector::Params ir_blob_detector_params;
  ir_blob_detector_params.filterByColor = true;
  ir_blob_detector_params.blobColor = 255;
  ir_blob_detector_params.filterByArea = true;
  ir_blob_detector_params.minArea = 10;
  ir_blob_detector_params.minThreshold = 150;
  ir_blob_detector_params.maxThreshold = 255;
  ir_blob_detector_params.thresholdStep = 20;
  ir_blob_detector_ = cv::SimpleBlobDetector::create(ir_blob_detector_params);

  // For depth.
  cv::SimpleBlobDetector::Params depth_blob_detector_params;
  depth_blob_detector_params.filterByArea = true;
  depth_blob_detector_params.minArea = 20;
  depth_blob_detector_params.minThreshold = 15;
  depth_blob_detector_params.maxThreshold = 175;
  depth_blob_detector_params.thresholdStep = 3;
  depth_blob_detector_ =
      cv::SimpleBlobDetector::create(depth_blob_detector_params);
}

std::pair<cv::Mat, cv::Mat> FrameAnalyzer::ConvertFramesToMats(
    const std::pair<rs2::depth_frame, rs2::video_frame>& frameset) {
  cv::Mat depth_mat, ir_mat;

  // 2cm when max_dist = 5.12m
  constexpr float unit_dist = MAX_DISTANCE / 256.0;
  float depth_conversion_scale = depth_scale_ / unit_dist;
  Mat(Size(848, 480), CV_16UC1, (void*)frameset.first.get_data(),
      Mat::AUTO_STEP)
      .convertTo(depth_mat, CV_8UC1, depth_conversion_scale);

  ir_mat = Mat(Size(848, 480), CV_8UC1, (void*)frameset.second.get_data(),
               Mat::AUTO_STEP);

  if (frameset_count_ > 500 && frameset_count_ < 520) {
    std::string debug_img_base_path = "imgs/" + std::to_string(frameset_count_);
    cv::imwrite(debug_img_base_path + "_ir.png", ir_mat);
    cv::imwrite(debug_img_base_path + "_depth.png", depth_mat);
  }

  return std::make_pair<cv::Mat, cv::Mat>(std::move(depth_mat),
                                          std::move(ir_mat));
}

AnalyzerOutput FrameAnalyzer::AnalyzeFrames(
    const std::pair<rs2::depth_frame, rs2::video_frame>& frameset) {
  auto mats = ConvertFramesToMats(frameset);

  return AnalyzeMats(mats);
}

AnalyzerOutput FrameAnalyzer::AnalyzeMats(
    const std::pair<cv::Mat, cv::Mat>& matset) {
  auto frame_start_ts = util::GetTimestampNanos();
  AnalyzerOutput depth_out, ir_out;

  std::thread depth_thread(&FrameAnalyzer::AnalyzeDepthMat, this,
                           std::ref(matset.first), &depth_out);
  std::thread ir_thread(&FrameAnalyzer::AnalyzeIRMat, this,
                        std::ref(matset.second), &ir_out);
  depth_thread.join();
  ir_thread.join();

  auto frame_end_ts = util::GetTimestampNanos();
  float frame_time = static_cast<float>(frame_end_ts - frame_start_ts);
  avg_frameset_analyze_time_ =
      (avg_frameset_analyze_time_ * frameset_count_ + frame_time / 1000) /
      (frameset_count_ + 1);
  frameset_count_++;
  return ir_out;
}

void FrameAnalyzer::AnalyzeDepthMat(cv::Mat depth_mat, AnalyzerOutput* output) {
  // Persistent depth.
  cv::Mat persistent_depth_img = depth_mat.clone();
  cv::Mat mask = cv::Mat::zeros(depth_mat.size(), depth_mat.type());
  for (int i = original_depth_frame_buffer_.size() - 1; i >= 0; --i) {
    cv::threshold(persistent_depth_img, mask, 1, 255, THRESH_BINARY_INV);
    original_depth_frame_buffer_[i].copyTo(persistent_depth_img, mask);
  }

  // Median.
  // cv::Mat median_depth = persistent_depth_img.clone();
  // cv::medianBlur(persistent_depth_img, median_depth, 3);

  /*
  // Threshold to remove background.
  const auto max_considered_dist = static_cast<uint8_t>(CONSIDERED_DISTANCE /
  unit_dist); cv::Mat bg_mask = original_depth_img.clone();
  cv::threshold(persistent_depth_img, bg_mask, max_considered_dist, 255,
  THRESH_BINARY_INV); persistent_depth_img.copyTo(output.marked_img, bg_mask);
  // output.marked_img = bg_mask.clone();
  */

  // std::vector<KeyPoint> keypoints;
  // depth_blob_detector_->detect(persistent_depth_img, keypoints);
  // cv::Mat im_with_keypoints;
  // cv::drawKeypoints(persistent_depth_img, keypoints, im_with_keypoints,
  //                   Scalar(0, 0, 255),
  //                   DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  cv::Mat detected_edges;
  cv::Canny(persistent_depth_img, detected_edges, 200, 300, 5);

  output->marked_img = detected_edges.clone();
  // Update stats and buffer etc, after each frame.
  original_depth_frame_buffer_.push_back(std::move(depth_mat));
  while (original_depth_frame_buffer_.size() > FRAME_BUFFER_SIZE) {
    original_depth_frame_buffer_.erase(original_depth_frame_buffer_.begin());
  }
}

void FrameAnalyzer::AnalyzeIRMat(cv::Mat ir_mat, AnalyzerOutput* output) {
  std::vector<KeyPoint> keypoints;
  ir_blob_detector_->detect(ir_mat, keypoints);
  cv::Mat im_with_keypoints;
  cv::drawKeypoints(ir_mat, keypoints, im_with_keypoints, Scalar(0, 0, 255),
                    DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  output->marked_img = im_with_keypoints.clone();
}

}  // namespace frame_analyzer