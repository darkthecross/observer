#include <librealsense2/rs.hpp>
#include <pcl/common/common_headers.h>
#include <sys/time.h>

namespace util {

float get_depth_scale(rs2::device dev);

rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);

std::tuple<int, int, int> RGB_Texture(rs2::video_frame texture, rs2::texture_coordinate Texture_XY);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr PCL_Conversion(const rs2::points& points, const rs2::video_frame& color);

long long int get_sys_msec();
}  // namespace util