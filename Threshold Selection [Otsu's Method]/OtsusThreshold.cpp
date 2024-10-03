/** Implementation file for finding Otsu's threshold
 *
 *  \file ipcv/otsus_threshold/OtsusThreshold.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 8 Sep 2018
 */

#include "OtsusThreshold.h"

#include <iostream>

#include "imgs/ipcv/utils/Utils.h"

using namespace std;

namespace ipcv {

/** Find Otsu's threshold for each channel of a 3-channel (color) image
 *
 *  \param[in] src          source cv::Mat of CV_8UC3
 *  \param[out] threshold   threshold values for each channel of a 3-channel
 *                          color image in cv::Vec3b
 */
bool OtsusThreshold(const cv::Mat& src, cv::Vec3b& threshold) {
  threshold = cv::Vec3b();

  // Create the matrices to store the histogram, pdf, cdf and flipped cdf
  cv::Mat src_h, src_pdf, src_cdf, flipped_cdf;

  // Create the matrix to hold our sigma values with the correct size
  cv::Mat goodnessMat = cv::Mat_<double>::zeros(3, 256);

  // Create our variables to store the points for max and min and goodness
  cv::Point minLoc, maxLoc, goodness;

  // Create our doubles to hold our claculated values
  double mu_k, mu_t, sigma_b, omega_k, DCmin, DCmax;

  // Compute the histogram, pdf and cdf
  Histogram(src, src_h);
  HistogramToPdf(src_h, src_pdf);
  HistogramToCdf(src_h, src_cdf);
  cv::flip(src_cdf, flipped_cdf, 1);

  // Index through each of the channels
  for (int channel = 0; channel < src_cdf.rows; channel++) {
    // Find the min and max for the cdf
    cv::minMaxLoc(flipped_cdf, NULL, NULL, &minLoc, NULL);
    cv::minMaxLoc(src_cdf, NULL, NULL, NULL, &maxLoc);

    // Set the values for the min and max based off the MinMaxLoc function
    DCmin = 255 - minLoc.x;
    DCmax = maxLoc.x;

    // Reset mu k and mu t for each interation of the channel loop due to it
    // being a summation.
    mu_t = 0;
    mu_k = 0;

    // Calculate mu t
    for (int i = 0; i < src_cdf.cols; i++) {
      mu_t += i * src_pdf.at<double>(channel, i);
    }
    // Calculate omega k, mu k, and sigma b. Then set the value of the goodness
    // matrix at that location to be sigma b.
    for (int i = DCmin; i < DCmax; i++) {
      omega_k = src_cdf.at<double>(channel, i);
      mu_k += i * src_pdf.at<double>(channel, i);
      sigma_b = ((mu_t * omega_k - mu_k) * (mu_t * omega_k - mu_k)) /
                (omega_k * (1 - omega_k));
      goodnessMat.at<double>(channel, i) = sigma_b;
    }

    // Find the max of the goodness matrix and add it to the threshold array
    cv::minMaxLoc(goodnessMat, NULL, NULL, NULL, &goodness);
    threshold[channel] = goodness.x;
  }

  return true;
}
}
