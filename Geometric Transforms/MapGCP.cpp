/** Implementation file for finding source image coordinates for a source-to-map
 *  remapping using ground control points
 *
 *  \file ipcv/geometric_transformation/MapGCP.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 15 Sep 2018
 */

#include "MapGCP.h"

#include <iostream>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

using namespace std;

namespace ipcv {

/** Find the source coordinates (map1, map2) for a ground control point
 *  derived mapping polynomial transformation
 *
 *  \param[in] src   source cv::Mat of CV_8UC3
 *  \param[in] map   map (target) cv::Mat of CV_8UC3
 *  \param[in] src_points
 *                   vector of cv::Points representing the ground control
 *                   points from the source image
 *  \param[in] map_points
 *                   vector of cv::Points representing the ground control
 *                   points from the map image
 *  \param[in] order  mapping polynomial order
 *                      EXAMPLES:
 *                        order = 1
 *                          a0*x^0*y^0 + a1*x^1*y^0 +
 *                          a2*x^0*y^1
 *                        order = 2
 *                          a0*x^0*y^0 + a1*x^1*y^0 + a2*x^2*y^0 +
 *                          a3*x^0*y^1 + a4*x^1*y^1 +
 *                          a5*x^0*y^2
 *                        order = 3
 *                          a0*x^0*y^0 + a1*x^1*y^0 + a2*x^2*y^0 + a3*x^3*y^0 +
 *                          a4*x^0*y^1 + a5*x^1*y^1 + a6*x^2*y^1 +
 *                          a7*x^0*y^2 + a8*x^1*y^2 +
 *                          a9*x^0*y^3
 *  \param[out] map1  cv::Mat of CV_32FC1 (size of the destination map)
 *                    containing the horizontal (x) coordinates at which to
 *                    resample the source data
 *  \param[out] map2  cv::Mat of CV_32FC1 (size of the destination map)
 *                    containing the vertical (y) coordinates at which to
 *                    resample the source data
 */
bool MapGCP(const cv::Mat src, const cv::Mat map,
            const vector<cv::Point> src_points,
            const vector<cv::Point> map_points, const int order, cv::Mat& map1,
            cv::Mat& map2) {
    // Construct X_Mat from the source points (target x and y coordinates)
  Eigen::MatrixXd X_Mat(map_points.size(), (order + 1) * (order + 2) / 2);

  for (size_t row_idx = 0; row_idx < map_points.size(); row_idx++) {
    double P_x = map_points[row_idx].x;
    double P_y = map_points[row_idx].y;
    int idx = 0;

    for (int i = 0; i <= order; i++) {
      for (int j = 0; j <= i; j++) {
        X_Mat(row_idx, idx) = std::pow(P_x, i - j) * std::pow(P_y, j);
        idx++;
      }
    }
  }

  // Construct Y_Mat from the source points (target x and y coordinates)
  Eigen::MatrixXd Y_Mat(src_points.size(), 2);
  for (size_t row_idx = 0; row_idx < src_points.size(); row_idx++) {
    Y_Mat(row_idx, 0) = src_points[row_idx].x;
    Y_Mat(row_idx, 1) = src_points[row_idx].y;
  }

  // Solve for the coefficients using the least-squares solution
  Eigen::MatrixXd AB =
      (X_Mat.transpose() * X_Mat).ldlt().solve(X_Mat.transpose() * Y_Mat);
  Eigen::VectorXd a = AB.col(0);  // Coefficients for x
  Eigen::VectorXd b = AB.col(1);  // Coefficients for y

  map1 = cv::Mat::zeros(map.size(), CV_32FC1);  // x-coordinate map
  map2 = cv::Mat::zeros(map.size(), CV_32FC1);  // y-coordinate map

  // Compute the mapped source coordinates for each pixel in the destination map
  for (int row = 0; row < map.rows; row++) {
    for (int col = 0; col < map.cols; col++) {
      double x = 0.0;
      double y = 0.0;
      int idx = 0;

      // Evaluate the polynomial for the current destination pixel (col, row)
      for (int i = 0; i <= order; i++) {
        for (int j = 0; j <= i; j++) {
          double term = std::pow(col, i - j) * std::pow(row, j);
          x += a(idx) * term;
          y += b(idx) * term;
          idx++;
        }
      }

      // Assign the computed x and y coordinates to the map
      map1.at<float>(row, col) = static_cast<float>(x);
      map2.at<float>(row, col) = static_cast<float>(y);
    }
  }

  return true;
}
}
