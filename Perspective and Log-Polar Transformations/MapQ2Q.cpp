/** Implementation file for mapping a source quad onto a target quad
 *
 *  \file ipcv/geometric_transformation/MapQ2Q.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 12 Sep 2020
 */

#include "MapQ2Q.h"

#include <iostream>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

using namespace std;

namespace ipcv {

/** Find the source coordinates (map1, map2) for a quad to quad mapping
 *
 *  \param[in] src       source cv::Mat of CV_8UC3
 *  \param[in] tgt       target cv::Mat of CV_8UC3
 *  \param[in] src_vertices
 *                       vertices cv::Point of the source quadrilateral (CW)
 *                       which is to be mapped to the target quadrilateral
 *  \param[in] tgt_vertices
 *                       vertices cv::Point of the target quadrilateral (CW)
 *                       into which the source quadrilateral is to be mapped
 *  \param[out] map1     cv::Mat of CV_32FC1 (size of the destination map)
 *                       containing the horizontal (x) coordinates at
 *                       which to resample the source data
 *  \param[out] map2     cv::Mat of CV_32FC1 (size of the destination map)
 *                       containing the vertical (y) coordinates at
 *                       which to resample the source data
 */
bool MapQ2Q(const cv::Mat src, const cv::Mat tgt,
            const vector<cv::Point> src_vertices,
            const vector<cv::Point> tgt_vertices, cv::Mat& map1,
            cv::Mat& map2) {

  // Initialize map1 and map2
  map1 = cv::Mat(tgt.size(), CV_32FC1);
  map2 = cv::Mat(tgt.size(), CV_32FC1);

  // Define matrices
  Eigen::MatrixXd map_mat(8, 8);
  Eigen::MatrixXd src_point(8, 1);
  Eigen::MatrixXd tgt_point(8, 1);
  Eigen::MatrixXd src_final(3, 1);
  Eigen::MatrixXd tgt_final(3, 1);

  // Fill the transformation matrix
  for (int i = 0; i < 4; i++) {
    map_mat(i, 0) = tgt_vertices[i].x;
    map_mat(i, 1) = tgt_vertices[i].y;
    map_mat(i, 2) = 1;
    map_mat(i, 3) = 0;
    map_mat(i, 4) = 0;
    map_mat(i, 5) = 0;
    map_mat(i, 6) = -tgt_vertices[i].x * src_vertices[i].x;
    map_mat(i, 7) = -tgt_vertices[i].y * src_vertices[i].x;

    map_mat(i + 4, 0) = 0;
    map_mat(i + 4, 1) = 0;
    map_mat(i + 4, 2) = 0;
    map_mat(i + 4, 3) = tgt_vertices[i].x;
    map_mat(i + 4, 4) = tgt_vertices[i].y;
    map_mat(i + 4, 5) = 1;
    map_mat(i + 4, 6) = -tgt_vertices[i].x * src_vertices[i].y;
    map_mat(i + 4, 7) = -tgt_vertices[i].y * src_vertices[i].y;
  }

  // Set up the source quad points matrix
  src_point << src_vertices[0].x,
  src_vertices[1].x, src_vertices[2].x, src_vertices[3].x, src_vertices[0].y,
  src_vertices[1].y, src_vertices[2].y, src_vertices[3].y;

  // Solve for the transformation coefficients
  tgt_point = map_mat.inverse() * src_point;

  // Build the perspective transformation matrix
  Eigen::MatrixXd P(3, 3);
  
  P << tgt_point(0), tgt_point(1), tgt_point(2), tgt_point(3), tgt_point(4),
      tgt_point(5), tgt_point(6), tgt_point(7), 1;

  // Loop through every pixel in the target image to find the corresponding source pixel
  for (int row_idx = 0; row_idx < tgt.rows; row_idx++) {
    for (int col_idx = 0; col_idx < tgt.cols; col_idx++) {
      
      // Create a point for the target pixel
      tgt_final << col_idx, row_idx, 1;
      
      // Transform the target point to the source using matrix P
      src_final << P * tgt_final;

      // Fill in final map with the solved result
      map1.at<float>(row_idx, col_idx) = static_cast<float>(src_final(0) / src_final(2));
      map2.at<float>(row_idx, col_idx) = static_cast<float>(src_final(1) / src_final(2));
    }
  }

  return true;
}
}
