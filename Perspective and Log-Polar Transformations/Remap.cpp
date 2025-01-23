/** Implementation file for remapping source values to map locations
 *
 *  \file ipcv/geometric_transformation/Remap.cpp
 *  \author Carl Salvaggio, Ph.D. (salvaggio@cis.rit.edu)
 *  \date 15 Sep 2018
 */

#include "Remap.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;

namespace ipcv {

/** Remap source values to the destination array at map1, map2 locations
 *
 *  \param[in] src            source cv::Mat of CV_8UC3
 *  \param[out] dst           destination cv::Mat of CV_8UC3 for remapped values
 *  \param[in] map1           cv::Mat of CV_32FC1 (size of the destination map)
 *                            containing the horizontal (x) coordinates at
 *                            which to resample the source data
 *  \param[in] map2           cv::Mat of CV_32FC1 (size of the destination map)
 *                            containing the vertical (y) coordinates at
 *                            which to resample the source data
 *  \param[in] interpolation  interpolation to be used for resampling
 *  \param[in] border_mode    border mode to be used for out-of-bounds pixels
 *  \param[in] border_value   border value to be used when constant border mode
 *                            is to be used
 */
bool Remap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map1,
           const cv::Mat& map2, const Interpolation interpolation,
           const BorderMode border_mode, const uint8_t border_value) {
  dst.create(map1.size(), src.type());

  double x, y;
  cv::Point2f remainder;

  // Nearest and Constant
  if (interpolation == Interpolation::NEAREST &&
      border_mode == BorderMode::CONSTANT) {
    for (int row_idx = 0; row_idx < dst.rows; row_idx++) {
      for (int col_idx = 0; col_idx < dst.cols; col_idx++) {
        x = map1.at<float>(row_idx, col_idx);
        y = map2.at<float>(row_idx, col_idx);

        if (x < 0 || x >= src.cols || y < 0 || y >= src.rows) {
          dst.at<cv::Vec3b>(row_idx, col_idx) =
              cv::Vec3b(border_value, border_value, border_value);
        } else {
          dst.at<cv::Vec3b>(row_idx, col_idx) =
              src.at<cv::Vec3b>(cv::Point((int)x, (int)y));
        }
      }
    }
  }

  // Nearest and Replicate
  if (interpolation == Interpolation::NEAREST &&
      border_mode == BorderMode::REPLICATE) {
    for (int row_idx = 0; row_idx < dst.rows; row_idx++) {
      for (int col_idx = 0; col_idx < dst.cols; col_idx++) {
        x = map1.at<float>(row_idx, col_idx);
        y = map2.at<float>(row_idx, col_idx);

        if (x < 0) {
          x = 0;
        } else if (x >= src.cols) {
          x = src.cols - 1;
        }

        if (y < 0) {
          y = 0;
        } else if (y >= src.rows) {
          y = src.rows - 1;
        }

        dst.at<cv::Vec3b>(row_idx, col_idx) =
            src.at<cv::Vec3b>(cv::Point((int)x, (int)y));
      }
    }
  }

  // Bilinear and Constant
  if (interpolation == Interpolation::LINEAR &&
      border_mode == BorderMode::CONSTANT) {
    for (int row_idx = 0; row_idx < dst.rows; row_idx++) {
      for (int col_idx = 0; col_idx < dst.cols; col_idx++) {
        x = map1.at<float>(row_idx, col_idx);
        y = map2.at<float>(row_idx, col_idx);

        if (x < 0 || x >= src.cols - 1 || y < 0 || y >= src.rows - 1) {
          dst.at<cv::Vec3b>(row_idx, col_idx) =
              cv::Vec3b(border_value, border_value, border_value);
        } else {
          int x1 = (int)floor(x);
          int y1 = (int)floor(y);
          int x2 = x1 + 1;
          int y2 = y1 + 1;

          remainder.x = x - x1;
          remainder.y = y - y1;

          for (int channel_idx = 0; channel_idx < 3; channel_idx++) {
            double top =
                (1 - remainder.x) * src.at<cv::Vec3b>(y1, x1)[channel_idx] +
                remainder.x * src.at<cv::Vec3b>(y1, x2)[channel_idx];
            double bottom =
                (1 - remainder.x) * src.at<cv::Vec3b>(y2, x1)[channel_idx] +
                remainder.x * src.at<cv::Vec3b>(y2, x2)[channel_idx];
            dst.at<cv::Vec3b>(row_idx, col_idx)[channel_idx] =
                (uchar)((1 - remainder.y) * top + remainder.y * bottom);
          }
        }
      }
    }
  }

  // Bilinear and Replicate
  if (interpolation == Interpolation::LINEAR &&
      border_mode == BorderMode::REPLICATE) {
    for (int row_idx = 0; row_idx < dst.rows; row_idx++) {
      for (int col_idx = 0; col_idx < dst.cols; col_idx++) {
        x = map1.at<float>(row_idx, col_idx);
        y = map2.at<float>(row_idx, col_idx);

        if (x < 0) {
          x = 0;
        } else if (x >= src.cols - 1) {
          x = src.cols - 2;
        }

        if (y < 0) {
          y = 0;
        } else if (y >= src.rows - 1) {
          y = src.rows - 2;
        }

        int x1 = (int)floor(x);
        int y1 = (int)floor(y);
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        remainder.x = x - x1;
        remainder.y = y - y1;

        for (int channel_idx = 0; channel_idx < 3; channel_idx++) {
          double top =
              (1 - remainder.x) * src.at<cv::Vec3b>(y1, x1)[channel_idx] +
              remainder.x * src.at<cv::Vec3b>(y1, x2)[channel_idx];
          double bottom =
              (1 - remainder.x) * src.at<cv::Vec3b>(y2, x1)[channel_idx] +
              remainder.x * src.at<cv::Vec3b>(y2, x2)[channel_idx];
          dst.at<cv::Vec3b>(row_idx, col_idx)[channel_idx] =
              (uchar)((1 - remainder.y) * top + remainder.y * bottom);
        }
      }
    }
  }

  return true;
}
}
