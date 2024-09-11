/** Implementation file for image enhancement using linear histogram statistics
 *
 *  \file ipcv/histogram_enhancement/LinearLut.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 3 Sep 2018
 * 
 *  \author Anthony Guarino (ag4933@rit.edu)
 *  \date 9 Sep 2024
 */

#include "LinearLut.h"

#include <iostream>

#include "imgs/ipcv/utils/Utils.h"

using namespace std;

namespace ipcv {

/** Create a 3-channel (color) LUT using linear histogram enhancement
 *
 *  \param[in] src          source cv::Mat of CV_8UC3
 *  \param[in] percentage   the total percentage to remove from the tails
 *                          of the histogram to find the extremes of the
 *                          linear enhancemnt function
 *  \param[out] lut         3-channel look up table in cv::Mat(3, 256)
 */
bool LinearLut(const cv::Mat& src, const int percentage, cv::Mat& lut) {
  // Propagate lut with 0s with the correct size.
  lut = cv::Mat::zeros(3, 256, CV_8UC1);

  // Define the histogram and CDF matrix
  cv::Mat h, cdf;

  // Compute Histogram and CDF for the source image
  ipcv::Histogram(src, h);
  ipcv::HistogramToCdf(h, cdf);

  /** Define Variables as 3 double vectors for each of the three color channels
   *
   *  \var max              Stores the maximum column for a given percentage.
   *  \var min              Stores the minimum column for a given percentage.
   *  \var slope            Stores the calculated slope using min and max.
   *  \var b                Uses slope intercept formula to find intercept.
   */
  vector<double> max, min, slope, b;
  max = {0, 0, 0};
  min = {0, 0, 0};
  slope = {0, 0, 0};
  b = {0, 0, 0};

  // Loop for each color channel
  for (int channel = 0; channel < cdf.rows; channel++) {
    // Find Max by starting at highest column (255) and going down until we hit
    // the target percentage (divided by 2 due to 2 tails)
    for (int i = cdf.cols - 1; i > 0; i--) {
      if (cdf.at<double>(channel, i) <= (1 - percentage / 200.0)) {
        max[channel] = i;
        break;
      }
    }

    // Find Min by starting at lowest column (0) and going up
    for (int i = 0; i < cdf.cols; i++) {
      if (cdf.at<double>(channel, i) >= (percentage / 200.0)) {
        min[channel] = i;
        break;
      }
    }
    // Calculate slope (y2-y1 / x2-x1)
    slope[channel] = 255.0 / (max[channel] - min[channel]);
    // Calculate y-intercept using slope intercept formula
    b[channel] = 255 - slope[channel] * max[channel];

    // Propagate Lookup Table
    for (int i = 0; i < cdf.cols; i++) {
      // Set value for any column below the minimum to 0
      if (i < min[channel]) {
        lut.at<uint8_t>(channel, i) = 0;
      }

      // Set value for any column below the minimum to 0
      else if (i > max[channel]) {
        lut.at<uint8_t>(channel, i) = 255;
      }

      // If the channel is between the max and the min, then set the value to be
      // the depended variable in the slope intercept formula
      else {
        lut.at<uint8_t>(channel, i) = (slope[channel] * i + b[channel]);
      }
    }
  }

  return true;
}
}
