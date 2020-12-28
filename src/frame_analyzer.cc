#include "frame_analyzer.h"

namespace frame_analyzer {

using namespace cv;

AnalyzerOutput DepthFrameAnalyzer::AnalyzeFrame(rs2::frame input_frame) {
  AnalyzerOutput output;
  Mat(Size(848, 480), CV_16UC1, (void*)input_frame.get_data(), Mat::AUTO_STEP)
      .convertTo(output.marked_img, CV_8UC1, 1.0/256.0);
  output.frame_count = ++frame_count_;
  return output;
}

}  // namespace frame_analyzer