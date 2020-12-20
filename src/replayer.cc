#include <chrono>
#include <ctime>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <string>

// #include "util.h"

using namespace cv;
using namespace std::chrono;

int64 GetTimestampMicros() {
  return duration_cast<microseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Arguments error! Usage: ./replayer xxx.bag" << std::endl;
  }
  const std::string input_file_name = argv[1];
  std::cout << "Replaying record from " << input_file_name << std::endl;

  rs2::config cfg;
  cfg.enable_device_from_file(input_file_name, false);
  rs2::pipeline pipe;
  // File will be opened in read mode at this point
  auto profile = pipe.start(cfg);

  if (!profile) {
    std::cout << "Realsense device error!" << std::endl;
    return -1;
  }

  // Aligh depth to infrared.
  rs2::align align(RS2_STREAM_INFRARED);

  rs2::device device = pipe.get_active_profile().get_device();

  if (device.as<rs2::playback>()) {
    std::cout << "playback device." << std::endl;
    device.as<rs2::playback>().resume();
  }

  int frame_count = 0;

  namedWindow("Display Image", WINDOW_AUTOSIZE);

  rs2::frameset frames;
  rs2::colorizer c;

  std::vector<Mat> color_frames(1000, Mat(Size(848, 480), CV_8UC1));
  std::vector<Mat> depth_frames(1000, Mat(Size(848, 480), CV_8UC3));

  int64 last_frame_time = GetTimestampMicros();
  int64 cur_time = last_frame_time;

  while (frame_count < 1000 && cur_time - last_frame_time < 5e5) {
    if (pipe.poll_for_frames(&frames)) {
      // Get processed aligned frame
      auto processed = align.process(frames);

      // Trying to get both other and aligned depth frames
      rs2::video_frame color_frame = processed.get_infrared_frame();
      rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

      auto colorized_depth = c.colorize(aligned_depth_frame);

      color_frames[frame_count] =
          Mat(Size(848, 480), CV_8UC1, (void*)color_frame.get_data(),
              Mat::AUTO_STEP)
              .clone();

      depth_frames[frame_count] =
          Mat(Size(848, 480), CV_8UC3, (void*)colorized_depth.get_data(),
              Mat::AUTO_STEP)
              .clone();

      last_frame_time = GetTimestampMicros();
      frame_count++;
    }
    cur_time = GetTimestampMicros();
  }
  pipe.stop();  // File will be closed at this point

  std::cout << "Total frame count: " << frame_count << std::endl;

  std::cout << "Got " << color_frames.size() << " RGB images, "
            << depth_frames.size() << " depth frames." << std::endl;

  for (auto i = 0; i < frame_count; ++i) {
    imshow("Display Image", color_frames[i]);
    waitKey(0);
  }

  for (auto i = 0; i < frame_count; ++i) {
    imshow("Display Image", depth_frames[i]);
    waitKey(0);
  }

  return 0;
}