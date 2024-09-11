/** Implementation file for image enhancement using histogram matching
 *
 *  \file ipcv/histogram_enhancement/MatchingLut.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 3 Sep 2018
 * 
 *  \author Anthony Guarino (ag4933@rit.edu)
 *  \date 9 Sep 2024
 */

#include "MatchingLut.h"

#include <iostream>

#include "imgs/ipcv/utils/Utils.h"

using namespace std;

namespace ipcv {

/** Create a 3-channel (color) LUT using histogram matching
 *
 *  \param[in] src   source cv::Mat of CV_8UC3
 *  \param[in] h     the histogram in cv:Mat(3, 256) that the
 *                   source is to be matched to
 *  \param[out] lut  3-channel look up table in cv::Mat(3, 256)
 */
bool MatchingLut(const cv::Mat& src, const cv::Mat& h, cv::Mat& lut) {
  // Propagate lut with 0s with the correct size.
  lut = cv::Mat::zeros(3, 256, CV_8UC1);

  // Define histogram and CDF matrix for the source image and a CDF matrix for
  // the target image
  cv::Mat src_h, src_cdf, h_cdf;

  // Define variables to store an index for the target histogram
  int h_idx;

  // Define varibale to store a value from the source image CDF and the target
  // image CDF
  double src_value;
  double h_value;

  // Compute Histogram for the source image and CDF for source image and the
  // target image
  ipcv::Histogram(src, src_h);
  ipcv::HistogramToCdf(src_h, src_cdf);
  ipcv::HistogramToCdf(h, h_cdf);

  // Loop through each of the 3 color channels
  for (int channel = 0; channel < src_cdf.rows; channel++) {
    // Reset the target image index and target image value
    h_idx = 0;
    h_value = 0;

    // Loop through each of the columns (0-255)
    for (int src_idx = 0; src_idx < src_cdf.cols; src_idx++) {
      // Store the value of the source CDF for the column and channel
      src_value = src_cdf.at<double>(channel, src_idx);

      // Check if the the target index is less than the total amount of columns
      // (255) and that the value of the target CDF is less than the value of
      // the source CDF
      while (h_idx < h_cdf.cols && h_value < src_value) {
        // If the previous statement is true, increase the index of the target
        // image CDF by one so that the value of the target image CDF will be of
        // the next column
        h_idx++;

        // Save the target value of the target image CDF
        h_value = h_cdf.at<double>(channel, h_idx);
      }
      // Once the conditions are false and the target image CDF is about equal
      // to the source image CDF or has reached the max colummn (255), the while
      // loop ends

      // Set the Lookup table value at the position that is being compared to be
      // equal to the index of the target image CDF
      lut.at<uint8_t>(channel, src_idx) = static_cast<uint8_t>(h_idx);
    }
  }

  return true;
}
}
