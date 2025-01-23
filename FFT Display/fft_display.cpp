#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "imgs/ipcv/utils/Utils.h"

using namespace std;
namespace po = boost::program_options;

void display(const cv::Mat& src, bool use_video) {
  // Convert to grayscale and pad the image
  cv::Mat gray;
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  int rows = cv::getOptimalDFTSize(gray.rows);
  int cols = cv::getOptimalDFTSize(gray.cols);

  cv::Mat padded;
  cv::copyMakeBorder(gray, padded, 0, rows - gray.rows, 0, cols - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  padded.convertTo(padded, CV_64F);

  // Compute DFT
  cv::Mat complex_image;
  cv::dft(padded, complex_image, cv::DFT_COMPLEX_OUTPUT);

  // Compute magnitude and phase
  cv::Mat planes[2];
  cv::split(complex_image, planes);
  cv::Mat mag, phase;
  cv::cartToPolar(planes[0], planes[1], mag, phase);

  // Shift the zero-frequency component to the center
  mag = ipcv::DftShift(mag);
  phase = ipcv::DftShift(phase);

  // Compute log magnitude for display
  cv::Mat log_mag;
  cv::log(mag + 1, log_mag);
  cv::Mat log_mag_display;
  cv::normalize(log_mag, log_mag_display, 0, 255, cv::NORM_MINMAX);
  log_mag_display.convertTo(log_mag_display, CV_8U);

  // Initialize matrices for reconstruction
  cv::Mat sum_components = cv::Mat::zeros(padded.size(), CV_64F);
  cv::Mat mask = cv::Mat::zeros(padded.size(), CV_64F);
  cv::Mat fourier_coefficient = cv::Mat::zeros(padded.size(), CV_64F);
  cv::Mat current_component, current_component_scaled;
  cv::Mat mag_copy = mag.clone();

  cv::VideoWriter video_writer;
  if (use_video) {
    // Initialize video writer with H.264 codec for MPEG-4
    video_writer.open("fft_display.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 30,
                      cv::Size(3 * log_mag.cols, 2 * log_mag.rows), true);
  }

  while (true) {
    int key = cv::waitKey(30);
    if (key == 27 || key == 'q' || key == 'Q') {
      break;
    }

    // Find the maximum magnitude location
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(mag_copy, &minVal, &maxVal, &minLoc, &maxLoc);

    if (maxVal == 0) {
      break;
    }

    // Create a mask for the current frequency
    mask.setTo(0);
    mask.at<double>(maxLoc.y, maxLoc.x) = 1.0;

    // Accumulate the Fourier coefficients for display
    fourier_coefficient += mask.mul(log_mag);

    // Mask the magnitude and phase
    cv::Mat current_mag = cv::Mat::zeros(mag.size(), mag.type());
    cv::Mat current_phase = cv::Mat::zeros(phase.size(), phase.type());
    current_mag.at<double>(maxLoc.y, maxLoc.x) = mag.at<double>(maxLoc.y, maxLoc.x);
    current_phase.at<double>(maxLoc.y, maxLoc.x) = phase.at<double>(maxLoc.y, maxLoc.x);

    // Shift back for inverse DFT
    current_mag = ipcv::DftShift(current_mag);
    current_phase = ipcv::DftShift(current_phase);

    // Convert polar to Cartesian coordinates
    cv::Mat real_part, imag_part;
    cv::polarToCart(current_mag, current_phase, real_part, imag_part);

    // Merge into complex image
    cv::Mat current_complex;
    cv::merge(std::vector<cv::Mat>{real_part, imag_part}, current_complex);

    // Inverse DFT to get current component in spatial domain
    cv::idft(current_complex, current_component, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    // Normalize current_component to [0, 1], scale it, and center around 128
    cv::Mat normalized_current_component;
    cv::normalize(current_component, normalized_current_component, 0, 1, cv::NORM_MINMAX);
    cv::Mat current_component_offset = (normalized_current_component * 20) + 128;

    // Accumulate the current component into `sum_components`
    sum_components += current_component;

    // Prepare images for display
    cv::Mat sum_components_display, fourier_coefficient_display;
    cv::normalize(sum_components, sum_components_display, 0, 255, cv::NORM_MINMAX);
    sum_components_display.convertTo(sum_components_display, CV_8U);

    cv::normalize(fourier_coefficient, fourier_coefficient_display, 0, 255, cv::NORM_MINMAX);
    fourier_coefficient_display.convertTo(fourier_coefficient_display, CV_8U);

    // Ensure `current_component_scaled` is initialized correctly
    cv::normalize(current_component, current_component_scaled, 0, 1, cv::NORM_MINMAX);
    current_component_scaled.convertTo(current_component_scaled, CV_8U, 255);

    // Create composite window
    cv::Mat composite_window(2 * log_mag.rows, 3 * log_mag.cols, CV_8U, cv::Scalar::all(0));

    // Place images in the composite window
    gray.convertTo(gray, CV_8U);
    gray.copyTo(composite_window(cv::Rect(0, 0, log_mag.cols, log_mag.rows)));
    log_mag_display.copyTo(composite_window(cv::Rect(log_mag.cols, 0, log_mag.cols, log_mag.rows)));
    current_component_offset.copyTo(composite_window(cv::Rect(2 * log_mag.cols, 0, log_mag.cols, log_mag.rows)));
    sum_components_display.copyTo(composite_window(cv::Rect(0, log_mag.rows, log_mag.cols, log_mag.rows)));
    fourier_coefficient_display.copyTo(composite_window(cv::Rect(log_mag.cols, log_mag.rows, log_mag.cols, log_mag.rows)));
    current_component_scaled.copyTo(composite_window(cv::Rect(2 * log_mag.cols, log_mag.rows, log_mag.cols, log_mag.rows)));

    cv::imshow("FFT Display", composite_window);

    // Ensure the composite window is in color before writing to video
    cv::Mat color_composite;
    cv::cvtColor(composite_window, color_composite, cv::COLOR_GRAY2BGR);

    if (use_video) {
      video_writer.write(color_composite);
    }

    // Zero out the used frequency in mag_copy
    mag_copy.at<double>(maxLoc.y, maxLoc.x) = 0;
  }

  if (use_video) {
    video_writer.release();
  }
}

int main(int argc, char* argv[]) {
  bool verbose, use_video;
  std::string src_filename;

  po::options_description visible_options("Options");
  visible_options.add_options()("help,h", "display this message")(
      "verbose,v", po::bool_switch(&verbose)->default_value(false),
      "verbose [default is silent]")(
      "record_video,r", po::bool_switch(&use_video)->default_value(false),
      "enable video recording");

  po::options_description hidden_options("Hidden Options");
  hidden_options.add_options()(
      "source-filename,i",
      po::value<std::string>(&src_filename)->default_value(""),
      "source filename");

  po::options_description all_options("All Options");
  all_options.add(visible_options).add(hidden_options);

  po::positional_options_description positional_options;
  positional_options.add("source-filename", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(all_options)
                .positional(positional_options)
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << "Usage: " << argv[0] << " [options] source-filename" << endl;
    cout << visible_options << endl;
    return EXIT_SUCCESS;
  }

  if (!boost::filesystem::exists(src_filename)) {
    cerr << "Provided source file does not exist" << endl;
    return EXIT_FAILURE;
  }

  cv::Mat src = cv::imread(src_filename, cv::IMREAD_COLOR);

  if (verbose) {
    cout << "Source filename: " << src_filename << endl;
    cout << "Size: " << src.size() << endl;
    cout << "Channels: " << src.channels() << endl;
  }
  display(src, use_video);
  cv::waitKey(0);

  return EXIT_SUCCESS;
}
