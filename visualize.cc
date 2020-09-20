#include <iostream>
#include <string>

// PCL Headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/io.h>
#include <pcl/visualization/cloud_viewer.h>

void Load_PCDFile(const std::string& file_name)
{
    // Generate object to store cloud in .pcd file
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudView (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::io::loadPCDFile (file_name, *cloudView); // Load .pcd File
    
    //==========================
    // Pointcloud Visualization
    //==========================
    // Create viewer object titled "Captured Frame"
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Captured Frame"));

    // Set background of viewer to black
    viewer->setBackgroundColor (0, 0, 0); 
    // Add generated point cloud and identify with string "Cloud"
    viewer->addPointCloud<pcl::PointXYZRGB> (cloudView, "Cloud");
    // Default size for rendered points
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Cloud");
    // Viewer Properties
    viewer->initCameraParameters();  // Camera Parameters for ease of viewing

    std::cout << endl;
    std::cout << "Press [Q] in viewer to continue. " << endl;
    
    viewer->spin(); // Allow user to rotate point cloud and view it
}

int main(int argc, char** argv) {
    if(argc!= 2) {
        std::cout << "Only one param is supported";
        return 0;
    }
    std::string file_name(argv[1]);
    Load_PCDFile(file_name);
    return 0;
}