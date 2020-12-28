#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "frame_analyzer.h"
#include "util.h"

DEFINE_string(input_file_name, "record_golden.bag", "path to recorded file.");

using namespace cv;

static std::string get_sensor_name(const rs2::sensor& sensor) {
  // Sensors support additional information, such as a human readable name
  if (sensor.supports(RS2_CAMERA_INFO_NAME))
    return sensor.get_info(RS2_CAMERA_INFO_NAME);
  else
    return "Unknown Sensor";
}

void EnqueueDepthFrame(const rs2::pipeline& p, rs2::align a, rs2::frame_queue q,
                       bool* running) {
  rs2::frameset frames;
  while (*running) {
    if (p.poll_for_frames(&frames)) {
      // Get processed aligned frame
      auto processed = a.process(frames);
      // Trying to get both other and aligned depth frames
      rs2::video_frame color = processed.get_depth_frame();
      q.enqueue(std::move(color));
      // Sleep for 7ms to reduce cpu usage.
      std::this_thread::sleep_for(std::chrono::microseconds(7));
    }
  }
  LOG(INFO) << "EqueneFrame thread joined.";
  return;
}

void ProcessDepthFrame(rs2::frame_queue q, cv::Mat* output_m,
                       frame_analyzer::DepthFrameAnalyzer* analyzer,
                       bool* running) {
  rs2::frame ir_frame;
  while (*running) {
    if (q.poll_for_frame(&ir_frame)) {
      auto output = analyzer->AnalyzeFrame(ir_frame);
      *output_m = output.marked_img.clone();
    }
  }
  LOG(INFO) << "ProcessFrame thread joined.";
  return;
}

int main(int argc, char** argv) {
  // Initialize log directory.
  google::InitGoogleLogging(argv[0]);
  if (!boost::filesystem::exists("logs")) {
    boost::filesystem::create_directory("logs");
  }
  FLAGS_log_dir = "logs";
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  rs2::config cfg;
  if (!FLAGS_input_file_name.empty()) {
    LOG(INFO) << "Replaying record from " << FLAGS_input_file_name;
    cfg.enable_device_from_file(FLAGS_input_file_name, false);
  }
  rs2::pipeline pipe;
  rs2::align align(RS2_STREAM_INFRARED);
  const auto CAPACITY = 10;
  rs2::frame_queue depth_queue(CAPACITY);
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

  if (!glfwInit()) {
    LOG(ERROR) << "Error initializing glfw! ";
    return -1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  GLFWwindow* window = glfwCreateWindow(848, 480, "Simple example", NULL, NULL);
  if (!window) {
    glfwTerminate();
    LOG(ERROR) << "Error initializing glfw window! ";
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  //  Initialise glew (must occur AFTER window creation or glew will error)
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    LOG(ERROR) << "GLEW initialisation error: " << glewGetErrorString(err);
    exit(-1);
  }
  LOG(INFO) << "GLEW okay - using version: " << glewGetString(GLEW_VERSION);

  cv::Mat disp_img;

  frame_analyzer::DepthFrameAnalyzer analyzer(depth_scale);
  util::InitOpenGL(848, 480);

  bool running = true;

  std::thread enqueue_frame_thread(EnqueueDepthFrame, std::ref(pipe), align,
                                   std::ref(depth_queue), &running);
  std::thread process_frame_thread(ProcessDepthFrame, std::ref(depth_queue),
                                   &disp_img, &analyzer, &running);

  while (!glfwWindowShouldClose(window))  // Application still alive?
  {
    util::DrawFrame(disp_img, 848, 480);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  running = false;

  LOG(INFO) << "Window closed.";

  enqueue_frame_thread.join();
  process_frame_thread.join();

  LOG(INFO) << "Execution completed. Processed "
            << analyzer.GetNumFramesProcessed() << " frames.";

  glfwDestroyWindow(window);
  glfwTerminate();

  pipe.stop();  // File will be closed at this point
  return 0;
}