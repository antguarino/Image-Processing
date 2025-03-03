#include <ctime>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "imgs/ipcv/geometric_transformation/GeometricTransformation.h"

using namespace std;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  bool verbose = false;
  string src_filename = "";
  string dst_filename = "";
  bool use_log = false;

  string interpolation_string = "nearest";
  int interpolation;

  po::options_description options("Options");
  options.add_options()("help,h", "display this message")(
      "verbose,v", po::bool_switch(&verbose), "verbose [default is silent]")(
      "source-filename,i", po::value<string>(&src_filename), "source filename")(
      "destination-filename,o", po::value<string>(&dst_filename),
      "destination filename [default is empty]")(
      "interpolation,t", po::value<string>(&interpolation_string),
      "interpolation (nearest|bilinear) [default is nearest]")(
      "use-log,l", po::bool_switch(&use_log),
      "use log-polar [default is polar]");

  po::positional_options_description positional_options;
  positional_options.add("source-filename", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(options)
                .positional(positional_options)
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << "Usage: " << argv[0] << " [options] source-filename" << endl;
    cout << options << endl;
    return EXIT_SUCCESS;
  }

  if (interpolation_string == "nearest") {
    interpolation = cv::INTER_NEAREST;
  } else if (interpolation_string == "bilinear") {
    interpolation = cv::INTER_LINEAR;
  } else {
    cerr << "*** ERROR *** ";
    cerr << "Provided interpolation is not supported" << endl;
    return EXIT_FAILURE;
  }

  if (!boost::filesystem::exists(src_filename)) {
    cerr << "*** ERROR *** ";
    cerr << "Provided source file does not exists" << endl;
    return EXIT_FAILURE;
  }

  cv::Mat src = cv::imread(src_filename, cv::IMREAD_COLOR);

  if (verbose) {
    cout << "Source filename: " << src_filename << endl;
    cout << "Size: " << src.size() << endl;
    cout << "Channels: " << src.channels() << endl;
    cout << "Interpolation: " << interpolation_string << endl;
    cout << "Use Log: " << use_log << endl;
    cout << "Destination filename: " << dst_filename << endl;
    
  }

  clock_t startTime = clock();

  bool status = false;
  cv::Mat map1;
  cv::Mat map2;
  status = ipcv::MapPolar(src, use_log, map1, map2);

  cv::Mat dst;
  cv::remap(src, dst, map1, map2, interpolation, cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0));

  clock_t endTime = clock();

  if (verbose) {
    cout << "Elapsed time: "
         << (endTime - startTime) / static_cast<double>(CLOCKS_PER_SEC)
         << " [s]" << endl;
  }

  if (status) {
    if (dst_filename.empty()) {
      cv::imshow(src_filename, src);
      cv::imshow(src_filename + " [RST]", dst);
      cv::waitKey(0);
    } else {
      cv::imwrite(dst_filename, dst);
    }
  } else {
    cerr << "*** ERROR *** ";
    cerr << "An error occurred while remapping image" << endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
