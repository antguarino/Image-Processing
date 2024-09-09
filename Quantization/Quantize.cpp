/** Implementation file for image quantization
 *
 *  \file ipcv/quantize/quantize.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 17 Mar 2018
 * 
 *  \author Anthony Guarino (ag4933@rit.edu)
 *  \date 1 Sep 2024
 *  \details Functions Uniform and Igs
 */

#include "Quantize.h"

#include <iostream>

using namespace std;

// ./bin/quantize -t uniform -l 7
// /home/ag4933/src/cpp/rit/imgs/ipcv/quantize/linear.tif

/** Perform uniform grey-level quantization on a color image
 *
 *  \param[in] src                 source cv::Mat of CV_8UC3
 *  \param[in] quantization_levels the number of levels to which to quantize
 *                                 the image
 *  \param[out] dst                destination cv:Mat of CV_8UC3
 */
void Uniform(const cv::Mat& src, const int& quantization_levels, cv::Mat& dst) {
  double bin_size = 256 / (static_cast<double>(quantization_levels));
  for (int y = 0; y < src.rows; y++) {    // Loop for each row
    for (int x = 0; x < src.cols; x++) {  // Loop for each column
      // Loop for each channel
      for (int channel = 0; channel < src.channels(); channel++) {
        // Pixel Value of the destination is equal to the quantized pixel value
        // of the source image
        dst.at<cv::Vec3b>(y, x)[channel] =
            src.at<cv::Vec3b>(y, x)[channel] / bin_size;
      }
    }
  }
}

/** Perform improved grey scale quantization on a color image
 *
 *  \param[in] src                 source cv::Mat of CV_8UC3
 *  \param[in] quantization_levels the number of levels to which to quantize
 *                                 the image
 *  \param[out] dst                destination cv:Mat of CV_8UC3
 */
void Igs(const cv::Mat& src, const int& quantization_levels, cv::Mat& dst) {
  // Error array for each of the channels
  double error[src.channels()] = {0, 0, 0};
  int value;
  double value_double;
  // Calculate bin_size
  int bin_size = 256 / (static_cast<double>(quantization_levels));
  for (int y = 0; y < src.rows; y++) {    // Loop for rows
    for (int x = 0; x < src.cols; x++) {  // Loop for columns
      // Loop for channels
      for (int channel = 0; channel < src.channels(); channel++) {
        // Set value and value_double = to the source image pixel value + error
        // The integer "value" and the double "value_double" will have different
        // values due to integer rounding
        value = value_double =
            src.at<cv::Vec3b>(y, x)[channel] + error[channel];
        value = value / bin_size * bin_size;
        // Error is going to be the difference of the actual pixel with error
        // and the rounded integerof the pixel with erorr
        error[channel] = value_double - value;
        // Pass the value to the destination image
        dst.at<cv::Vec3b>(y, x)[channel] = value / bin_size;
      }
    }
  }
}

namespace ipcv {

bool Quantize(const cv::Mat& src, const int quantization_levels,
              const QuantizationType quantization_type, cv::Mat& dst) {
  dst.create(src.size(), src.type());

  switch (quantization_type) {
    case QuantizationType::uniform:
      Uniform(src, quantization_levels, dst);
      break;
    case QuantizationType::igs:
      Igs(src, quantization_levels, dst);
      break;
    default:
      cerr << "Specified quantization type is unsupported" << endl;
      return false;
  }

  return true;
}
}
