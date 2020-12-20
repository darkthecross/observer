#include <gflags/gflags.h>
#include <glog/logging.h>
#include <boost/filesystem.hpp>
#include <chrono>
#include <ctime>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "imgui.h"
#include "imgui_impl_glfw.h"

DEFINE_string(input_file_name, "record_golden.bag", "path to recorded file.");

using namespace cv;
using namespace std::chrono;

int64 GetTimestampMicros() {
  return duration_cast<microseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

int64 GetTimestampNanos() {
  return duration_cast<nanoseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if(!boost::filesystem::exists("logs")) {
    boost::filesystem::create_directory("logs");
  }
  FLAGS_log_dir = "logs";

  LOG(INFO) << "Replaying record from " << FLAGS_input_file_name;

  rs2::config cfg;
  cfg.enable_device_from_file(FLAGS_input_file_name, false);
  rs2::pipeline pipe;
  rs2::align align(RS2_STREAM_INFRARED);

  std::vector<Mat> color_frames(1000, Mat(Size(848, 480), CV_8UC1));
  std::vector<Mat> depth_frames(1000, Mat(Size(848, 480), CV_8UC3));
  size_t color_frame_count = 0;
  size_t depth_frame_count = 0;

  std::mutex color_frame_mutex;
  std::mutex depth_frame_mutex;

  rs2::colorizer c;

  const auto CAPACITY = 10;
  rs2::frame_queue depth_queue(CAPACITY);
  rs2::frame_queue ir_queue(CAPACITY);

  auto profile = pipe.start(cfg);

  if (!profile) {
    LOG(INFO) << "Realsense device error!";
    return -1;
  }

  int64 last_frame_time = GetTimestampMicros();
  int64 cur_time = last_frame_time;

  rs2::frameset frames;

  size_t frames_count = 0;
  bool stopped = false;

  cv::namedWindow("Depth", WINDOW_AUTOSIZE);
  cv::namedWindow("IR", WINDOW_AUTOSIZE);

  std::thread depth_t([&]() {
    while (!stopped) {
      rs2::frame f;
      if (depth_queue.poll_for_frame(&f)) {
        const std::lock_guard<std::mutex> lock(depth_frame_mutex);
        auto colorized_depth = c.colorize(f);
        depth_frames[depth_frame_count] =
            Mat(Size(848, 480), CV_8UC3, (void*)colorized_depth.get_data(),
                Mat::AUTO_STEP)
                .clone();
        imshow("Depth", depth_frames[depth_frame_count]);
        waitKey(1);
        depth_frame_count++;
      }
    }
  });

  std::thread ir_t([&]() {
    while (!stopped) {
      rs2::frame f;
      if (ir_queue.poll_for_frame(&f)) {
        const std::lock_guard<std::mutex> lock(color_frame_mutex);
        color_frames[color_frame_count] =
            Mat(Size(848, 480), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP)
                .clone();
        imshow("IR", color_frames[color_frame_count]);
        waitKey(1);
        color_frame_count++;
      }
    }
  });

  while (cur_time - last_frame_time < 5e5) {
    if (pipe.poll_for_frames(&frames)) {
      // Get processed aligned frame
      auto processed = align.process(frames);

      // Trying to get both other and aligned depth frames
      rs2::video_frame color_frame = processed.get_infrared_frame();
      rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

      ir_queue.enqueue(color_frame);
      depth_queue.enqueue(aligned_depth_frame);

      last_frame_time = GetTimestampMicros();
      frames_count++;
    }
    cur_time = GetTimestampMicros();
  }

  pipe.stop();  // File will be closed at this point
  stopped = true;
  depth_t.join();
  ir_t.join();

  LOG(INFO) << "color_frame_count: " << color_frame_count
            << " depth_frame_count: " << depth_frame_count;

  return 0;
}