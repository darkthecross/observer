#include <ctime>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <string>

int main(int argc, char** argv) {
  std::time_t result = std::time(nullptr);
  std::stringstream ss;
  ss << "record_";
  ss << result;
  ss << ".bag";

  std::cout << "Will record to " << ss.str() << ". " << std::endl;

  rs2::pipeline pipe;
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
  cfg.enable_stream(RS2_STREAM_INFRARED, 848, 480, RS2_FORMAT_Y8, 90);
  const std::string record_file_name = ss.str();
  cfg.enable_record_to_file(record_file_name);

  pipe.start(cfg);  // File will be opened in write mode at this point
  for (int i = 0; i < 900; i++) {
    auto frames = pipe.wait_for_frames();
    result = std::time(nullptr);
    if (i % 50 == 0) {
      std::cout << "Frame " << i << " recorded at "
                << std::asctime(std::localtime(&result)) << result
                << " seconds since the Epoch\n";
    }
  }
  pipe.stop();  // File will be closed at this point
  return 0;
}