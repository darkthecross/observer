#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "frame_analyzer.h"
#include "util.h"

using namespace cv;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  rs2_intrinsics intr;
  intr.fx = 422.617;
  intr.fy = 422.617;
  intr.ppx = 425.752;
  intr.ppy = 240.141;
  intr.model = RS2_DISTORTION_BROWN_CONRADY;
  for(int i = 0; i<5; ++i) {
    intr.coeffs[i] = 0.0;
  }
  frame_analyzer::FrameAnalyzer analyzer(intr, 1e-3);

  cv::namedWindow("dev");

  for (size_t i = 501; i < 520; ++i) {
    std::string path_base = "imgs/" + std::to_string(i);
    cv::Mat depth_mat =
        cv::imread(path_base + "_depth.png", cv::IMREAD_GRAYSCALE);
    cv::Mat depth_mat_u16;
    depth_mat.convertTo(depth_mat_u16, CV_16UC1, 20.0);
    cv::Mat ir_mat = cv::imread(path_base + "_ir.png", cv::IMREAD_GRAYSCALE);

    auto output = analyzer.CorePipeline(std::make_pair<cv::Mat, cv::Mat>(
        std::move(depth_mat_u16), std::move(ir_mat)));

    LOG(INFO) << "avg time: " << analyzer.GetAverageFramesetAnalyzeTime()
              << "us.";

    cv::imshow("dev", output.marked_img);
    cv::waitKey(0);
  }
  return 0;
}