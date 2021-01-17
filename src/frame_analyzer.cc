#include "frame_analyzer.h"

#include <librealsense2/rsutil.h>

#include <iostream>
#include <string>
#include <thread>

#include "gpu.h"
#include "util.h"

namespace frame_analyzer {

using namespace cv;

namespace {

std::vector<FeaturePoint> FilterByRadius(const std::vector<FeaturePoint>& fps,
                                         Mat depth_mat, float depth_scale,
                                         const rs2_intrinsics& intr) {
  std::vector<FeaturePoint> results;
  for (const auto& fp : fps) {
    float center[3], border[3];
    float center_pixel[2] = {fp.y, fp.x};
    float center_depth =
        static_cast<float>(depth_mat.at<ushort>(fp.y, fp.x) * depth_scale);
    rs2_deproject_pixel_to_point(center, &intr, center_pixel, center_depth);

    float border_pixel[2] = {fp.y + fp.r, fp.x + fp.r};
    rs2_deproject_pixel_to_point(border, &intr, border_pixel, center_depth);

    float dist[3] = {border[0] - center[0], border[1] - center[1],
                     border[2] - center[2]};

    float radius =
        sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]);

    if (radius > 0.01 && radius < 0.1) {
      results.push_back(fp);
    }
  }
  return results;
}

std::vector<FeaturePoint> FilterByBackground(
    const std::vector<FeaturePoint>& fps, Mat depth_mat) {
  std::vector<FeaturePoint> results;
  for (const auto& fp : fps) {
    Mat mask(depth_mat.size(), CV_8UC1, Scalar(0));
    cv::circle(mask, Point(fp.x, fp.y), fp.r * 1.5, Scalar(255), 2);

    float to_be_averaged = 0.0;
    int num_pixels = 0;
    for (int xx = fp.x - fp.r * 1.8; xx < fp.x + fp.r * 1.8; ++xx) {
      for (int yy = fp.y - fp.r * 1.8; yy < fp.y + fp.r * 1.8; ++yy) {
        if (xx < 0 || xx >= depth_mat.size().width || yy < 0 ||
            yy >= depth_mat.size().height)
          continue;
        if (depth_mat.at<ushort>(yy, xx) == 0) continue;
        if (mask.at<uchar>(yy, xx) == 0) continue;
        to_be_averaged += depth_mat.at<ushort>(yy, xx);
        ++num_pixels;
        // std::cout << "to_be_averaged: " << to_be_averaged
        //           << " num_pixels: " << num_pixels << std::endl;
      }
    }
    float avg_bg = to_be_averaged / num_pixels;
    if (abs(avg_bg - static_cast<float>(depth_mat.at<ushort>(fp.y, fp.x))) >
        20) {
      results.push_back(fp);
    }
  }
  return results;
}

}  // namespace

FrameAnalyzer::FrameAnalyzer(rs2_intrinsics intrinsics, float depth_scale) {
  depth_scale_ = depth_scale;
  depth_intrinsics_ = intrinsics;

  // For IR, we detect a light blob which is quite obvious.
  cv::SimpleBlobDetector::Params ir_blob_detector_params;
  ir_blob_detector_params.filterByColor = true;
  ir_blob_detector_params.blobColor = 255;
  ir_blob_detector_params.filterByArea = true;
  ir_blob_detector_params.minArea = 5;
  ir_blob_detector_params.minThreshold = 150;
  ir_blob_detector_params.maxThreshold = 255;
  ir_blob_detector_params.thresholdStep = 30;
  // Filter by Circularity
  ir_blob_detector_params.filterByCircularity = false;
  ir_blob_detector_params.minCircularity = 0.1;
  // Filter by Convexity
  ir_blob_detector_params.filterByConvexity = false;
  ir_blob_detector_params.minConvexity = 0.87;
  // Filter by Inertia
  ir_blob_detector_params.filterByInertia = false;
  ir_blob_detector_params.minInertiaRatio = 0.01;
  ir_blob_detector_ = cv::SimpleBlobDetector::create(ir_blob_detector_params);
}

std::pair<cv::Mat, cv::Mat> FrameAnalyzer::ConvertFramesToMats(
    const std::pair<rs2::depth_frame, rs2::video_frame>& frameset) {
  cv::Mat depth_mat, ir_mat;

  depth_mat = Mat(Size(848, 480), CV_16UC1, (void*)frameset.first.get_data(),
                  Mat::AUTO_STEP);

  ir_mat = Mat(Size(848, 480), CV_8UC1, (void*)frameset.second.get_data(),
               Mat::AUTO_STEP);

  // if (frameset_count_ > 500 && frameset_count_ < 520) {
  //   std::string debug_img_base_path = "imgs/" +
  //   std::to_string(frameset_count_); constexpr float unit_dist = MAX_DISTANCE
  //   / 256.0;
  // float depth_conversion_scale = depth_scale_ / unit_dist;
  // cv::Mat depth_mat_8u(depth_mat.size(), CV_8UC1);
  // depth_mat.convertTo(depth_mat_8u, CV_8UC1, depth_conversion_scale);
  //   cv::imwrite(debug_img_base_path + "_ir.png", ir_mat);
  //   cv::imwrite(debug_img_base_path + "_depth.png", depth_mat_8u);
  // }

  return std::make_pair<cv::Mat, cv::Mat>(std::move(depth_mat),
                                          std::move(ir_mat));
}

AnalyzerOutput FrameAnalyzer::CorePipeline(
    const std::pair<cv::Mat, cv::Mat>& matset) {
  IROutput ir_out = AnalyzeIRMat(matset.second);
  DepthOutput depth_out = AnalyzeDepthMat(matset.first, ir_out);
  return ProcessFeaturePoints(depth_out);
}

AnalyzerOutput FrameAnalyzer::AnalyzeFrames(
    const std::pair<rs2::depth_frame, rs2::video_frame>& frameset) {
  auto frame_start_ts = util::GetTimestampNanos();

  auto mats = ConvertFramesToMats(frameset);

  auto analyzer_out = FrameAnalyzer::CorePipeline(mats);

  auto frame_end_ts = util::GetTimestampNanos();
  float frame_time = static_cast<float>(frame_end_ts - frame_start_ts);
  avg_frameset_analyze_time_ =
      (avg_frameset_analyze_time_ * frameset_count_ + frame_time / 1000) /
      (frameset_count_ + 1);
  frameset_count_++;

  return analyzer_out;
}

IROutput FrameAnalyzer::AnalyzeIRMat(cv::Mat ir_mat) {
  IROutput output;

  cv::Mat blurred_img;
  cv::medianBlur(ir_mat, blurred_img, 5);

  std::vector<KeyPoint> keypoints;
  ir_blob_detector_->detect(ir_mat, keypoints);
  cv::Mat im_with_keypoints;
  cv::drawKeypoints(blurred_img, keypoints, im_with_keypoints,
                    Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  for (const auto& pts : keypoints) {
    output.feature_points.push_back(FeaturePoint(pts.pt.x, pts.pt.y, pts.size));
  }
  return output;
}

DepthOutput FrameAnalyzer::AnalyzeDepthMat(cv::Mat depth_mat,
                                           const IROutput& ir_output) {
  // Persistent depth.
  // cv::Mat persistent_depth_img = depth_mat.clone();
  // cv::Mat mask = cv::Mat::zeros(depth_mat.size(), depth_mat.type());
  // cv::Mat mask_8u = cv::Mat(depth_mat.size(), CV_8UC1);
  // for (int i = original_depth_frame_buffer_.size() - 1; i >= 0; --i) {
  //   cv::threshold(persistent_depth_img, mask, 1, 65535, THRESH_BINARY_INV);
  //   mask.convertTo(mask_8u, CV_8UC1);
  //   original_depth_frame_buffer_[i].copyTo(persistent_depth_img, mask_8u);
  // }

  DepthOutput output;

  // 2cm when max_dist = 5.12m
  constexpr float unit_dist = MAX_DISTANCE / 256.0;
  float depth_conversion_scale = depth_scale_ / unit_dist;
  cv::Mat depth_mat_8u(depth_mat.size(), CV_8UC1);
  depth_mat.convertTo(depth_mat_8u, CV_8UC1, depth_conversion_scale);

  std::vector<FeaturePoint> fps;
  fps = FilterByRadius(ir_output.feature_points, depth_mat, depth_scale_,
                       depth_intrinsics_);
  output.filtered_feature_points = FilterByBackground(fps, depth_mat);

  // Update buffers.
  original_depth_frame_buffer_.push_back(std::move(depth_mat));
  while (original_depth_frame_buffer_.size() > FRAME_BUFFER_SIZE) {
    original_depth_frame_buffer_.erase(original_depth_frame_buffer_.begin());
  }

  return output;
}

AnalyzerOutput FrameAnalyzer::ProcessFeaturePoints(
    const DepthOutput& depth_output) {
  AnalyzerOutput output;

  feature_points_buffer_.push_back(depth_output.filtered_feature_points);
  while (feature_points_buffer_.size() > FRAME_BUFFER_SIZE) {
    feature_points_buffer_.erase(feature_points_buffer_.begin());
  }

  // 2cm when max_dist = 5.12m
  constexpr float unit_dist = MAX_DISTANCE / 256.0;
  float depth_conversion_scale = depth_scale_ / unit_dist;
  cv::Mat depth_mat_8u(original_depth_frame_buffer_.back().size(), CV_8UC1);
  original_depth_frame_buffer_.back().convertTo(depth_mat_8u, CV_8UC1,
                                                depth_conversion_scale);

  cv::Mat im_with_keypoints = depth_mat_8u.clone();
  for (const auto& pts : depth_output.filtered_feature_points) {
    cv::circle(im_with_keypoints, Point(pts.x, pts.y), pts.r, Scalar(255), 1);
  }

  output.marked_img = im_with_keypoints.clone();

  return output;
}

}  // namespace frame_analyzer