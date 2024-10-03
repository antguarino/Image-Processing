#include <ctime>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/plot.hpp>

#include "imgs/ipcv/otsus_threshold/OtsusThreshold.h"
#include "imgs/ipcv/utils/Utils.h"
#include "imgs/plot/plot.h"

using namespace std;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  bool verbose = false;
  string src_filename = "";
  string dst_filename = "";

  po::options_description options("Options");
  options.add_options()("help,h", "display this message")(
      "verbose,v", po::bool_switch(&verbose), "verbose [default is silent]")(
      "source-filename,i", po::value<string>(&src_filename), "source filename")(
      "destination-filename,o", po::value<string>(&dst_filename),
      "destination filename");

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

  if (!boost::filesystem::exists(src_filename)) {
    cerr << "Provided source file does not exists" << endl;
    return EXIT_FAILURE;
  }

  cv::Mat src = cv::imread(src_filename, cv::IMREAD_COLOR);

  if (verbose) {
    cout << "Source filename: " << src_filename << endl;
    cout << "Size: " << src.size() << endl;
    cout << "Channels: " << src.channels() << endl;
    cout << "Destination filename: " << dst_filename << endl;
  }

  cv::Vec3b threshold;

  clock_t startTime = clock();

  ipcv::OtsusThreshold(src, threshold);

  clock_t endTime = clock();

  if (verbose) {
    cout << "Elapsed time: "
         << (endTime - startTime) / static_cast<double>(CLOCKS_PER_SEC)
         << " [s]" << endl;
  }

  if (verbose) {
    cout << "Threshold values = ";
    cout << threshold << endl;
  }

  cv::Mat lut;
  lut.create(3, 256, CV_8UC1);
  for (int b = 0; b < 3; b++) {
    for (int dc = 0; dc < 256; dc++) {
      lut.at<uint8_t>(b, dc) = (dc <= threshold[b]) ? 0 : 255;
    }
  }

  cv::Mat dst;
  ipcv::ApplyLut(src, lut, dst);

  if (dst_filename.empty()) {
    cv::imshow(src_filename, src);
    cv::imshow(src_filename + " [Thresholded]", dst);
    cv::waitKey(0);
  } else {
    cv::imwrite(dst_filename, dst);
  }

  // Create Matrices for Histogram and PDF
  cv::Mat h, pdf;

  plot::plot2d::Params params;

  // Create Vetor for all titles
  vector<string> title = {"Blue Channel", "Green Channel", "Red Channel"};

  // Create label for each axis
  params.set_x_label("Digital Count");
  params.set_y_label("PDF Value");

  // Compute Histogram and PDF
  ipcv::Histogram(src, h);
  ipcv::HistogramToPdf(h, pdf);

  // Create Vectors for the data to be stored in
  std::vector<double> x(256);
  std::vector<double> y(256);

  // Index through each channel for the individual graphs
  for (int channel = 0; channel < pdf.rows; channel++) {
    // Set title based on the channel by indexing the vector made previously
    params.set_title(title[channel]);
    // Fill the x and y vectors with the digital count and pdf values
    // respectivly
    for (int i = 0; i < pdf.cols; i++) {
      x.at(i) = i;
      y.at(i) = pdf.at<double>(channel, i);
    }
    // Plot the line indicating the threshold
    params.set_xvline(threshold[channel]);

    // Plot the graph using the vectors and the set parameters
    plot::plot2d::Plot2d(x, y, params);
  }

  return EXIT_SUCCESS;
}
