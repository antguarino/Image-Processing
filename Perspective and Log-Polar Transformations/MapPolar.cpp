/** Interface file for finding map coordinates for MapPolar
 *
 *  \file ipcv/geometric_transformation/MapPolar.h
 *  \author Anthony Guarino (ag4933@rit.edu)
 *  \date 8 Oct 2024
 */

#include "MapPolar.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

using namespace std;

namespace ipcv {

/** Find the source coordinates (map1, map2) for a polar or log-polar transformation
 *
 *  \param[in] src       source cv::Mat of CV_8UC3
 *  \param[in] use_log   boolean to toggle log-polar transformation
 *  \param[out] map1     cv::Mat of CV_32FC1 (size of the destination map)
 *                       containing the horizontal (x) coordinates to
 *                       resample the source data
 *  \param[out] map2     cv::Mat of CV_32FC1 (size of the destination map)
 *                       containing the vertical (y) coordinates to
 *                       resample the source data
 */

// Initialize the map1 and map2 matrices
bool MapPolar(const cv::Mat src, const bool use_log, cv::Mat& map1,
              cv::Mat& map2) {
  map1.create(src.size(), CV_32FC1);
  map2.create(src.size(), CV_32FC1);

  // Calculate parameters for the number of sectors and rings, and image center and max radius (rho)
  int numSectors = src.cols;
  int numRings = src.rows;
  double centerX = src.cols / 2.0;
  double centerY = src.rows / 2.0;
  float rho_max = sqrt(centerX * centerX + centerY * centerY);

  vector<double> cosValues(numSectors);
  vector<double> sinValues(numSectors);

  // Check if using polar or log-polar
  if (use_log) {
    rho_max = log(rho_max + 1.0);
  }

  // Set the first row of the map with the center coordinates
  for (int i = 0; i < numSectors; i++) {
    double theta = 2.0 * M_PI * i / numSectors;
    cosValues[i] = cos(theta);
    sinValues[i] = sin(theta);

    map1.at<float>(0, i) = centerX;
    map2.at<float>(0, i) = centerY;
  }

  // Loop through each ring to calculate the x and y coordinates
  for (int j = 1; j < numRings; j++) {
    // Calculate the radius
    float rho =
        use_log ? (exp(j * rho_max / numRings) - 1) : (j * rho_max / numRings);

    for (int i = 0; i < numSectors; i++) {
      float x = static_cast<float>(centerX + rho * cosValues[i]);
      float y = static_cast<float>(centerY + rho * sinValues[i]);

      // Fill maps with solved values
      if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
        map1.at<float>(j, i) = x;
        map2.at<float>(j, i) = y;
      }

      else {
        map1.at<float>(j, i) = -1;
        map2.at<float>(j, i) = -1;
      }
    }
  }

  return true;
}

}
