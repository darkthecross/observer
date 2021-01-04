#include <glog/logging.h>
#include <gflags/gflags.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "frame_analyzer.h"
#include "util.h"

using namespace cv;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);


  frame_analyzer::FrameAnalyzer analyzer;

  cv::namedWindow("dev");

  for (size_t i = 501; i < 520; ++i) {
    std::string path_base = "imgs/" + std::to_string(i);
    cv::Mat depth_mat =
        cv::imread(path_base + "_depth.png", cv::IMREAD_GRAYSCALE);
    cv::Mat ir_mat = cv::imread(path_base + "_ir.png", cv::IMREAD_GRAYSCALE);

    auto output = analyzer.AnalyzeMats(std::make_pair<cv::Mat, cv::Mat>(
        std::move(depth_mat), std::move(ir_mat)));

    LOG(INFO) << "avg time: " << analyzer.GetAverageFramesetAnalyzeTime()
              << "us.";

    cv::imshow("dev", output.marked_img);
    cv::waitKey(0);
  }
  return 0;
}