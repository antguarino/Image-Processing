/** Implementation file for finding map coordinates for an RST transformation
 *
 *  \file ipcv/geometric_transformation/MapRST.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 26 Sep 2019
 */

#include "MapRST.h"

#include <iostream>

#include <Eigen/Dense>

using namespace std;

namespace ipcv {

/** Find the map coordinates (map1, map2) for an RST transformation
 *
 *  \param[in] src           source cv::Mat of CV_8UC3
 *  \param[in] angle         rotation angle (CCW) [radians]
 *  \param[in] scale_x       horizontal scale
 *  \param[in] scale_y       vertical scale
 *  \param[in] translation_x horizontal translation [+ right]
 *  \param[in] translation_y vertical translation [+ up]
 *  \param[out] map1         cv::Mat of CV_32FC1 (size of the destination map)
 *                           containing the horizontal (x) coordinates at
 *                           which to resample the source data
 *  \param[out] map2         cv::Mat of CV_32FC1 (size of the destination map)
 *                           containing the vertical (y) coordinates at
 *                           which to resample the source data
 */
bool MapRST(const cv::Mat src, const double angle, const double scale_x,
            const double scale_y, const double translation_x,
            const double translation_y, cv::Mat& map1, cv::Mat& map2) {
  // Create Matricies
  Eigen::Matrix3d rotation;
  Eigen::Matrix3d scale;
  Eigen::Matrix3d translation;
  Eigen::Matrix3d affine;

  // Stream in values for matricies
  rotation << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1;
  scale << 1 / scale_x, 0, 0, 0, 1 / scale_y, 0, 0, 0, 1;
  translation << 1, 0, -translation_x, 0, 1, translation_y, 0, 0, 1;
  affine = rotation * scale * translation;

  // Calculate corners
  Eigen::Vector3d TopLeft;
  TopLeft << -src.cols / 2, src.rows / 2, 1;
  Eigen::Vector3d BottomLeft;
  BottomLeft << -src.cols / 2, -src.rows / 2, 1;
  Eigen::Vector3d TopRight;
  TopRight << src.cols / 2, src.rows / 2, 1;
  Eigen::Vector3d BottomRight;
  BottomRight << src.cols / 2, -src.rows / 2, 1;

  // Calculate new transformed corners
  Eigen::Vector3d NTopLeft = affine.inverse() * TopLeft;
  Eigen::Vector3d NBottomLeft = affine.inverse() * BottomLeft;
  Eigen::Vector3d NTopRight = affine.inverse() * TopRight;
  Eigen::Vector3d NBottomRight = affine.inverse() * BottomRight;

  // Find size of rows and columns
  int rows = static_cast<int>(std::ceil(std::max(
      abs(NTopLeft(1) - NBottomRight(1)), abs(NTopRight(1) - NBottomLeft(1)))));
  int cols = static_cast<int>(std::ceil(std::max(
      abs(NTopLeft(0) - NBottomRight(0)), abs(NTopRight(0) - NBottomLeft(0)))));

  // Make maps the correct size based on size of transformed rows and columns
  map1.create(rows, cols, CV_32FC1);
  map2.create(rows, cols, CV_32FC1);

  // Fill map1 and map2 with the values from the location at the source
  for (int row_idx = 0; row_idx < rows; row_idx++) {
    for (int col_idx = 0; col_idx < cols; col_idx++) {
      Eigen::Vector3d dst_point;
      dst_point << col_idx - cols / 2, row_idx - rows / 2, 1;
      Eigen::Vector3d src_point = affine * dst_point;

      map1.at<float>(row_idx, col_idx) =
          static_cast<float>(src_point(0) + src.cols / 2);
      map2.at<float>(row_idx, col_idx) =
          static_cast<float>(src_point(1) + src.rows / 2);
    }
  }

  return true;
}
}