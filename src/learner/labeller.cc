#include <gflags/gflags.h>
#include <glog/logging.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

using namespace std;

DEFINE_string(img_dir, "/home/darkthecross/Documents/observer/data/examples/",
              "dir to dump examples.");

int main(int argc, char** argv) {
  // Initialize log directory.
  google::InitGoogleLogging(argv[0]);
  if (!boost::filesystem::exists("logs")) {
    boost::filesystem::create_directory("logs");
  }
  FLAGS_log_dir = "logs";
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::map<std::string, std::vector<std::string>> all_files;

  boost::filesystem::directory_iterator end_itr;
  for (boost::filesystem::directory_iterator i(FLAGS_img_dir); i != end_itr;
       ++i) {
    // Skip if not a file
    if (!boost::filesystem::is_regular_file(i->status())) continue;

    // File matches, store it
    vector<string> strs;
    boost::split(strs, i->path().filename().string(), boost::is_any_of("_"));
    all_files[strs[0]].push_back(strs[1]);
  }

  auto count_not_labelled = 0;
  for (auto ex : all_files) {
    if (ex.second.size() == 2) {
      count_not_labelled++;
    }
  }

  LOG(INFO) << "Read completed, " << count_not_labelled
            << " files not labelled.";
  return 0;
}