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

constexpr size_t FRAME_QUEUE_CAPACITY = 10;

using FramesetQueue = std::queue<std::pair<rs2::depth_frame, rs2::video_frame>>;

using namespace cv;

static std::string get_sensor_name(const rs2::sensor& sensor) {
  // Sensors support additional information, such as a human readable name
  if (sensor.supports(RS2_CAMERA_INFO_NAME))
    return sensor.get_info(RS2_CAMERA_INFO_NAME);
  else
    return "Unknown Sensor";
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action,
                         int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
}

void EnqueueFrames(const rs2::pipeline& p, rs2::align a, FramesetQueue* q,
                   std::atomic<bool>* running) {
  rs2::frameset frames;
  while (*running) {
    if (p.poll_for_frames(&frames)) {
      // Get processed aligned frame
      auto processed = a.process(frames);
      // Trying to get both other and aligned depth frames
      rs2::video_frame ir_frame = processed.get_infrared_frame();
      rs2::depth_frame depth_frame = processed.get_depth_frame();
      q->push(std::make_pair<rs2::depth_frame, rs2::video_frame>(
          std::move(depth_frame), std::move(ir_frame)));
      while (q->size() > FRAME_QUEUE_CAPACITY) {
        LOG(WARNING) << "Frameset dropped from queue.";
        q->pop();
      }
      // Sleep for 100us to reduce cpu usage.
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }
  LOG(INFO) << "EqueneFrame thread joined.";
  return;
}

void ProcessFrames(FramesetQueue* q, cv::Mat* output_m,
                   frame_analyzer::FrameAnalyzer* analyzer,
                   std::atomic<bool>* running) {
  while (*running) {
    if (!q->empty()) {
      auto frame_set = q->front();
      q->pop();
      auto output = analyzer->AnalyzeFrames(frame_set);
      if (analyzer->GetNumFramesetsProcessed() % 500 == 0) {
        LOG(INFO) << "Avg processing time: "
                  << analyzer->GetAverageFramesetAnalyzeTime() << "us";
      }
      *output_m = output.marked_img.clone();
      // Sleep for 100us to reduce cpu usage.
      std::this_thread::sleep_for(std::chrono::microseconds(100));
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
  if (FLAGS_input_file_name.empty()) {
    // Start camera stream.
    LOG(INFO) << "Starting camera stream...";
    cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
    cfg.enable_stream(RS2_STREAM_INFRARED, 848, 480, RS2_FORMAT_Y8, 90);
  } else {
    // Read from bag for debug/dev purpose.
    LOG(INFO) << "Replaying record from " << FLAGS_input_file_name;
    cfg.enable_device_from_file(FLAGS_input_file_name, true);
  }
  rs2::pipeline pipe;
  rs2::align align(RS2_STREAM_INFRARED);
  // rs2::frame_queue depth_queue(CAPACITY);

  FramesetQueue frameset_queue;

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
  GLFWwindow* window = glfwCreateWindow(848, 480, "Simple example",
                                        glfwGetPrimaryMonitor(), NULL);
  if (!window) {
    glfwTerminate();
    LOG(ERROR) << "Error initializing glfw window! ";
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  glfwSetKeyCallback(window, key_callback);

  //  Initialise glew (must occur AFTER window creation or glew will error)
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    LOG(ERROR) << "GLEW initialisation error: " << glewGetErrorString(err);
    exit(-1);
  }
  LOG(INFO) << "GLEW okay - using version: " << glewGetString(GLEW_VERSION);

  cv::Mat disp_img;

  frame_analyzer::FrameAnalyzer analyzer(depth_scale);
  util::InitOpenGL(848, 480);

  std::atomic<bool> running;
  running = true;

  std::thread enqueue_frame_thread(EnqueueFrames, std::ref(pipe), align,
                                   &frameset_queue, &running);
  std::thread process_frame_thread(ProcessFrames, &frameset_queue, &disp_img,
                                   &analyzer, &running);

  while (!glfwWindowShouldClose(window))  // Application still alive?
  {
    util::DrawFrame(disp_img, 848, 480);
    glfwSwapBuffers(window);
    glfwPollEvents();
    // Limit fps to 110.
    std::this_thread::sleep_for(std::chrono::milliseconds(9));
  }

  running = false;

  LOG(INFO) << "Window closed.";

  enqueue_frame_thread.join();
  process_frame_thread.join();

  LOG(INFO) << "Execution completed. Processed "
            << analyzer.GetNumFramesetsProcessed() << " frames.";

  glfwDestroyWindow(window);
  glfwTerminate();

  pipe.stop();  // File will be closed at this point
  return 0;
}