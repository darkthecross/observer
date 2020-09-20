#include <iostream>
#include <librealsense2/rs.hpp>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>
#include <pcl/io/io.h>

#include "util.h"

int main(int argc, char** argv) {
    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start();
    float depth_scale = util::get_depth_scale(profile.get_device());
    // Get color stream.
    //If there is no color stream, choose to align depth to another stream
    rs2_stream align_to = util::find_stream_to_align(profile.get_streams());
    // Aligh depth to color.
    rs2::align align(align_to);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr newCloud (new pcl::PointCloud<pcl::PointXYZRGB>);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;

    for(int i = 0; i<10; i++) {
        // Using the align object, we block the application until a frameset is available
        rs2::frameset frameset = pipe.wait_for_frames();

        // rs2::pipeline::wait_for_frames() can replace the device it uses in case of device error or disconnection.
        // Since rs2::align is aligning depth to some other stream, we need to make sure that the stream was not changed
        //  after the call to wait_for_frames();
        if (util::profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            profile = pipe.get_active_profile();
            align_to = util::find_stream_to_align(profile.get_streams());
            align = rs2::align(align_to);
            depth_scale = util::get_depth_scale(profile.get_device());
        }

        //Get processed aligned frame
        auto processed = align.process(frameset);

        // Trying to get both other and aligned depth frames
        rs2::video_frame other_frame = processed.first(align_to);
        rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

        // Map Color texture to each point
        pc.map_to(other_frame);

        // Generate Point Cloud
        auto points = pc.calculate(aligned_depth_frame);

        // Convert generated Point Cloud to PCL Formatting
        auto cloud = util::PCL_Conversion(points, other_frame);

        //========================================
        // Filter PointCloud (PassThrough Method)
        //========================================
        pcl::PassThrough<pcl::PointXYZRGB> Cloud_Filter; // Create the filtering object
        Cloud_Filter.setInputCloud (cloud);           // Input generated cloud to filter
        Cloud_Filter.setFilterFieldName ("z");        // Set field name to Z-coordinate
        Cloud_Filter.setFilterLimits (0.0, 1.0);      // Set accepted interval values
        Cloud_Filter.filter (*newCloud);              // Filtered Cloud Outputted
        
        std::string cloudFile = "Captured_Frame" + std::to_string(i) + ".pcd";
        
        //==============================
        // Write PC to .pcd File Format
        //==============================
        // Take Cloud Data and write to .PCD File Format
        std::cout << "Generating PCD Point Cloud File... " << std::endl;
        pcl::io::savePCDFileASCII(cloudFile, *cloud); // Input cloud to be saved to .pcd
        std::cout << cloudFile << " successfully generated. " << std::endl; 
    }

    return 0; 
}