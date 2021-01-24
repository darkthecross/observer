#include "frame_analyzer.h"

#include <librealsense2/rsutil.h>

#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "gpu.h"
#include "util.h"

namespace frame_analyzer {

using namespace cv;

namespace {

WorldObject FeaturePointToObj(const FeaturePoint& fp, Mat depth_mat,
                              float depth_scale, const rs2_intrinsics& intr) {
  float center[3], border[3];

  float center_depth = static_cast<float>(
      depth_mat.at<ushort>(fp.coord[1], fp.coord[0]) * depth_scale);

  float center_pixel[2] = {fp.coord[1], fp.coord[0]};
  rs2_deproject_pixel_to_point(center, &intr, center_pixel, center_depth);

  float border_pixel[2] = {fp.coord[1] + fp.r, fp.coord[0] + fp.r};
  rs2_deproject_pixel_to_point(border, &intr, border_pixel, center_depth);

  float dist[3] = {border[0] - center[0], border[1] - center[1],
                   border[2] - center[2]};
  WorldObject obj;
  obj.position << center[0], center[1], center[2];
  obj.radius = sqrt(dist[0] * dist[0] + dist[1] * dist[1] + dist[2] * dist[2]);
  obj.velocity = Eigen::Vector3f::Zero();
  obj.feature = fp;
  return obj;
}

std::vector<FeaturePoint> FilterByRadius(const std::vector<FeaturePoint>& fps,
                                         Mat depth_mat, float depth_scale,
                                         const rs2_intrinsics& intr) {
  std::vector<FeaturePoint> results;
  for (const auto& fp : fps) {
    auto obj = FeaturePointToObj(fp, depth_mat, depth_scale, intr);

    if (obj.radius > 0.01 && obj.radius < 0.1) {
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
    cv::circle(mask, Point(fp.coord[0], fp.coord[1]), fp.r * 1.5, Scalar(255),
               2);

    float to_be_averaged = 0.0;
    int num_pixels = 0;
    for (int xx = fp.coord[0] - fp.r * 1.8; xx < fp.coord[0] + fp.r * 1.8;
         ++xx) {
      for (int yy = fp.coord[1] - fp.r * 1.8; yy < fp.coord[1] + fp.r * 1.8;
           ++yy) {
        if (xx < 0 || xx >= depth_mat.size().width || yy < 0 ||
            yy >= depth_mat.size().height)
          continue;
        if (depth_mat.at<ushort>(yy, xx) == 0) continue;
        if (mask.at<uchar>(yy, xx) == 0) continue;
        to_be_averaged += depth_mat.at<ushort>(yy, xx);
        ++num_pixels;
      }
    }
    float avg_bg = to_be_averaged / num_pixels;
    if (avg_bg -
            static_cast<float>(depth_mat.at<ushort>(fp.coord[1], fp.coord[0])) >
        20.0) {
      results.push_back(fp);
    }
  }
  return results;
}

std::vector<FeaturePoint> FilterByDepthDiff(
    const std::vector<FeaturePoint>& fps, Mat depth_mat) {
  std::vector<FeaturePoint> results;
  for (const auto& fp : fps) {
    Rect roi(fp.coord[0] - fp.r * 1.5, fp.coord[1] - fp.r * 1.5, fp.r * 3,
             fp.r * 3);
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width >= depth_mat.size().width ||
        roi.y + roi.height >= depth_mat.size().height)
      continue;
    Mat roi_mat = Mat(depth_mat, roi).clone();
    Mat resized_mat;
    resize(roi_mat, resized_mat, Size(30, 30));
    double target_avg = 0.0, bg_avg_min = std::numeric_limits<double>::max();
    for (int ii = 0; ii < 3; ii++) {
      for (int jj = 0; jj < 3; jj++) {
        Rect small_roi(ii * 10, jj * 10, 10, 10);
        Mat ref_small(resized_mat, small_roi);
        double avg = mean(ref_small, ref_small > 0).val[0];
        if (ii != 1 || jj != 1) {
          bg_avg_min = min(avg, bg_avg_min);
        } else {
          target_avg = avg;
        }
      }
    }

    if (target_avg < bg_avg_min - 20.0) {
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
  ir_blob_detector_params.minArea = 15;
  ir_blob_detector_params.minThreshold = 150;
  ir_blob_detector_params.maxThreshold = 255;
  ir_blob_detector_params.thresholdStep = 50;
  // Filter by Circularity
  ir_blob_detector_params.filterByCircularity = true;
  ir_blob_detector_params.minCircularity = 0.1;
  // Filter by Convexity
  ir_blob_detector_params.filterByConvexity = true;
  ir_blob_detector_params.minConvexity = 0.87;
  // Filter by Inertia
  ir_blob_detector_params.filterByInertia = false;
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

  frame_ts_buffer_.push_back(frameset.first.get_timestamp());
  while (frame_ts_buffer_.size() > FRAME_BUFFER_SIZE) {
    frame_ts_buffer_.erase(frame_ts_buffer_.begin());
  }

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
  // auto fps1 = FilterByBackground(fps, depth_mat);
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

  // correspondence[i] = j => tracked_objects_ i corresponds to tmp_world_obj j.
  std::vector<WorldObject> tmp_world_obj;
  for (const auto& fp : depth_output.filtered_feature_points) {
    tmp_world_obj.push_back(
        FeaturePointToObj(fp, original_depth_frame_buffer_.back(), depth_scale_,
                          depth_intrinsics_));
  }

  constexpr float MAX_RADIUS_ERROR_RATIO = 2.0;
  // 0.2m / 0.01s = 20m/s
  constexpr float MAX_DIST_ERROR = 0.2;

  float timestep = 0.0;
  int frame_ts_buffer_size = frame_ts_buffer_.size();
  if (frame_ts_buffer_size >= 2) {
    timestep = frame_ts_buffer_[frame_ts_buffer_size - 1] -
               frame_ts_buffer_[frame_ts_buffer_size - 2];
  }

  // Match tracked_objects_ to closest tmp_world_obj.
  std::vector<std::pair<int, float>> mapping_with_dist_error;
  std::map<int, std::vector<int>> tmp_to_world_map;
  for (int world_obj_id = 0; world_obj_id < tracked_objects_.size();
       ++world_obj_id) {
    const WorldObject& world_obj = tracked_objects_[world_obj_id];
    float min_dist = std::numeric_limits<float>::max();
    int match_id = -1;
    for (int tmp_obj_id = 0; tmp_obj_id < tmp_world_obj.size(); ++tmp_obj_id) {
      const auto& tmp_obj = tmp_world_obj[tmp_obj_id];
      auto projected_position =
          world_obj.position + world_obj.velocity * timestep;
      auto diff_vec = tmp_obj.position - projected_position;
      auto diff_vec_norm = diff_vec.norm();
      if (diff_vec_norm > MAX_DIST_ERROR) continue;
      if (world_obj.radius < tmp_obj.radius / MAX_RADIUS_ERROR_RATIO ||
          world_obj.radius > tmp_obj.radius * MAX_RADIUS_ERROR_RATIO)
        continue;
      min_dist = min(min_dist, diff_vec_norm);
      if (min_dist == diff_vec_norm) match_id = tmp_obj_id;
    }
    mapping_with_dist_error.push_back(std::make_pair(match_id, min_dist));
    tmp_to_world_map[match_id].push_back(world_obj_id);
  }

  // Remove duplicates -- keep only the closest match.
  for (const auto& tmp_to_world_pair : tmp_to_world_map) {
    if (tmp_to_world_pair.second.size() <= 1) continue;
    float min_dist = std::numeric_limits<float>::max();
    int selected_id = -1;
    for (const auto& world_id : tmp_to_world_pair.second) {
      if (min_dist > mapping_with_dist_error[world_id].second) {
        min_dist = mapping_with_dist_error[world_id].second;
        selected_id = world_id;
      }
    }
    for (const auto& world_id : tmp_to_world_pair.second) {
      if (world_id != selected_id) {
        mapping_with_dist_error[world_id] =
            std::make_pair(-1, std::numeric_limits<float>::max());
      }
    }
  }

  for (int world_obj_id = 0; world_obj_id < tracked_objects_.size();
       ++world_obj_id) {
    if (mapping_with_dist_error[world_obj_id].first != -1) {
      // For objects with a match, update velocity and belief.
      auto p =
          tmp_world_obj[mapping_with_dist_error[world_obj_id].first].position -
          tracked_objects_[world_obj_id].position;
      tracked_objects_[world_obj_id].position =
          tmp_world_obj[mapping_with_dist_error[world_obj_id].first].position;
      tracked_objects_[world_obj_id].feature =
          tmp_world_obj[mapping_with_dist_error[world_obj_id].first].feature;
      // TODO(fanmx): This velocity estimation could be inaccurate.
      if (timestep > 0) {
        tracked_objects_[world_obj_id].velocity = p / timestep;
      }
      tracked_objects_[world_obj_id].belief =
          tracked_objects_[world_obj_id].belief +
          (1.0 - tracked_objects_[world_obj_id].belief) * 0.5;
    } else {
      // Decrease belief for unmatched objects.
      tracked_objects_[world_obj_id].belief *= 0.5;
    }
  }
  // Add new unmatched objects.
  for (int tmp_obj_id = 0; tmp_obj_id < tmp_world_obj.size(); ++tmp_obj_id) {
    if (tmp_to_world_map.find(tmp_obj_id) == tmp_to_world_map.end()) {
      tracked_objects_.push_back(tmp_world_obj[tmp_obj_id]);
      tracked_objects_.back().belief = 0.5;
    }
  }
  // Remove tracked_obj with really low belief.
  auto predicate = [](const WorldObject& w) { return w.belief < 0.3; };
  tracked_objects_.erase(std::remove_if(tracked_objects_.begin(),
                                        tracked_objects_.end(), predicate),
                         tracked_objects_.end());

  // 2cm when max_dist = 5.12m
  constexpr float unit_dist = MAX_DISTANCE / 256.0;
  float depth_conversion_scale = depth_scale_ / unit_dist;
  cv::Mat depth_mat_8u(original_depth_frame_buffer_.back().size(), CV_8UC1);
  original_depth_frame_buffer_.back().convertTo(depth_mat_8u, CV_8UC1,
                                                depth_conversion_scale);
  cv::Mat depth_mat_rgb;
  cv::cvtColor(depth_mat_8u, depth_mat_rgb, COLOR_GRAY2BGR);

  cv::Mat im_with_keypoints = depth_mat_rgb.clone();
  for (const auto& wp : tracked_objects_) {
    if (wp.belief < 0.95) continue;
    const auto& feature = wp.feature;
    cv::circle(im_with_keypoints, Point(feature.coord[0], feature.coord[1]),
               feature.r, Scalar(0, 0, 255), 2);
  }
  for (const auto& fp : depth_output.filtered_feature_points) {
    cv::circle(im_with_keypoints, Point(fp.coord[0], fp.coord[1]), fp.r,
               Scalar(255, 0, 0), 1);
  }

  output.marked_img = im_with_keypoints.clone();

  return output;
}

}  // namespace frame_analyzer