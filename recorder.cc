#include <iostream>
#include <librealsense2/rs.hpp>
#include <string>
#include <ctime>

int main(int argc, char** argv) {
    rs2::config cfg;
    std::time_t result = std::time(nullptr);
    std::stringstream ss;
    ss << "record_";
    ss << result;
    ss << ".bag";
    const std::string record_file_name = ss.str();
    cfg.enable_record_to_file(record_file_name);
    rs2::pipeline pipe;
    pipe.start(cfg); //File will be opened in write mode at this point
    for (int i = 0; i < 30; i++)
    {
        auto frames = pipe.wait_for_frames();
        result = std::time(nullptr);
        std::cout << "Frame " << i << " recorded at " << std::asctime(std::localtime(&result))
                << result << " seconds since the Epoch\n";
    }
    pipe.stop(); //File will be closed at this point
    return 0;
}