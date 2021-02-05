#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <chrono>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

DEFINE_string(
    input_file_name,
    "/home/darkthecross/Documents/observer/data/records/record_golden.bag",
    "path to recorded file.");

DEFINE_string(output_dir,
              "/home/darkthecross/Documents/observer/data/examples/",
              "dir to dump examples.");

using namespace cv;
using namespace std::chrono;

int64 GetTimestampNanos() {
  return duration_cast<nanoseconds>(
             high_resolution_clock::now().time_since_epoch())
      .count();
}

int main(int argc, char** argv) {
  // Initialize log directory.
  google::InitGoogleLogging(argv[0]);
  if (!boost::filesystem::exists("logs")) {
    boost::filesystem::create_directory("logs");
  }
  FLAGS_log_dir = "logs";
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize output dir.
  if (!boost::filesystem::exists(FLAGS_output_dir)) {
    boost::filesystem::create_directory(FLAGS_output_dir);
  }

  rs2::config cfg;
  if (FLAGS_input_file_name.empty()) {
    // Start camera stream.
    LOG(INFO) << "Starting camera stream...";
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
    cfg.enable_stream(RS2_STREAM_INFRARED, 848, 480, RS2_FORMAT_Y8, 90);
  } else {
    // Read from bag for debug/dev purpose.
    LOG(INFO) << "Replaying record from " << FLAGS_input_file_name;
    cfg.enable_device_from_file(FLAGS_input_file_name, false);
  }
  rs2::pipeline pipe;
  rs2::align align(RS2_STREAM_INFRARED);

  auto profile = pipe.start(cfg);
  if (!profile) {
    LOG(ERROR) << "Realsense device error! ";
    return -1;
  }
  LOG(INFO) << "Pipeline started.";

  float depth_scale;
  auto sensors = profile.get_device().query_sensors();
  int index = 0;
  // We can now iterate the sensors and print their names
  for (rs2::sensor sensor : sensors) {
    if (sensor.supports(RS2_CAMERA_INFO_NAME) &&
        std::strcmp(sensor.get_info(RS2_CAMERA_INFO_NAME), "Stereo Module") ==
            0) {
      depth_scale = sensor.as<rs2::depth_sensor>().get_depth_scale();
    }
  }
  LOG(INFO) << "Depth scale: " << depth_scale;

  auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH);
  rs2_intrinsics depth_stream_intrinsics =
      depth_stream.as<rs2::video_stream_profile>().get_intrinsics();

  rs2::frameset frames;

  int frame_count = 0;
  while (true) {
    frames = pipe.wait_for_frames();
    // Get processed aligned frame
    auto processed = align.process(frames);
    // Trying to get both other and aligned depth frames
    rs2::video_frame ir_frame = processed.get_infrared_frame();
    rs2::depth_frame depth_frame = processed.get_depth_frame();

    auto cur_us = GetTimestampNanos();
    cv::Mat depth_mat, ir_mat;
    depth_mat = Mat(Size(848, 480), CV_16UC1, (void*)depth_frame.get_data(),
                    Mat::AUTO_STEP);
    ir_mat = Mat(Size(848, 480), CV_8UC1, (void*)ir_frame.get_data(),
                 Mat::AUTO_STEP);
    const std::string depth_img_name =
        std::string(FLAGS_output_dir) + std::to_string(cur_us) + "_depth.png";
    imwrite(depth_img_name, depth_mat);
    const std::string ir_img_name =
        std::string(FLAGS_output_dir) + std::to_string(cur_us) + "_ir.png";
    imwrite(ir_img_name, ir_mat);
    // Sleep for 100us to reduce cpu usage.
    frame_count++;
    if (frame_count % 50 == 0)
      LOG(INFO) << "Processed " << frame_count << " frames.";
  }

  pipe.stop();  // File will be closed at this point
  return 0;
}