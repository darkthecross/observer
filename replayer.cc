#include <iostream>
#include <librealsense2/rs.hpp>
#include <string>
#include <ctime>

#include <boost/thread/thread.hpp>
#include <opencv2/opencv.hpp>

#include "util.h"

using namespace cv;

int main(int argc, char** argv) {
    const std::string input_file_name = argv[1];
    std::cout << "Replaying record from " << input_file_name << std::endl;

    rs2::config cfg;
    cfg.enable_device_from_file(input_file_name, false);
    rs2::pipeline pipe;
    auto profile = pipe.start(cfg); //File will be opened in read mode at this point

    rs2_stream align_to = util::find_stream_to_align(profile.get_streams());
    // Aligh depth to color.
    rs2::align align(align_to);

    rs2::device device = pipe.get_active_profile().get_device();

    if(device.as<rs2::playback>()) {
        std::cout << "playback device." << std::endl;
        device.as<rs2::playback>().resume();
    }

    if(device.as<rs2::recorder>()) {
        std::cout << "recorder device." << std::endl;
    }

    int frame_count = 0;

    namedWindow("Display Image", WINDOW_AUTOSIZE );

    rs2::frameset frames;

    std::vector<Mat> color_frames;

    while (frame_count < 400)
    {
        if(pipe.poll_for_frames(&frames)) {
            frame_count ++;

            std::cout << "Started processing frame " << frame_count << " at " << util::get_sys_msec() << std::endl;

            //Get processed aligned frame
            auto processed = align.process(frames);

            // Trying to get both other and aligned depth frames
            rs2::video_frame other_frame = processed.first(align_to);
            rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

            Mat color(Size(640, 480), CV_8UC3, (void*)other_frame.get_data(), Mat::AUTO_STEP);
            color_frames.push_back(color.clone());

            std::cout << "Completed processing frame " << frame_count << " at " << util::get_sys_msec() << std::endl;
        }
    }

    pipe.stop(); //File will be closed at this point

    for(auto m : color_frames) {
        imshow("Display Image", m);
        waitKey(0);
    }

    waitKey(0);

    return 0;
}