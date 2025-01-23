#include "BilateralFilter.h"
#include <opencv2/opencv.hpp>
#include <cmath>

namespace ipcv {

/** Bilateral filter an image
 *
 *  \param[in] src             source cv::Mat of CV_8UC3
 *  \param[out] dst            destination cv::Mat of ddepth type
 *  \param[in] sigma_distance  standard deviation of distance/closeness filter
 *  \param[in] sigma_range     standard deviation of range/similarity filter
 *  \param[in] radius          radius of the bilateral filter (if negative, use
 *                             twice the standard deviation of the distance/
 *                             closeness filter)
 *  \param[in] border_mode     pixel extrapolation method
 *  \param[in] border_value    value to use for constant border mode
 */
bool BilateralFilter(const cv::Mat& src, cv::Mat& dst,
                     const double sigma_distance, const double sigma_range,
                     const int radius, const BorderMode border_mode,
                     uint8_t border_value) {
  // Set up dst with the same size and type as src
  dst.create(src.size(), src.type());

  float filtered_value, weight_sum;

  int new_radius =
      (radius <= 0) ? static_cast<int>(2 * sigma_distance) : radius;

  cv::Mat src_border, src_lab, dst_lab = cv::Mat::zeros(src.size(), CV_32FC3);

  // Expand the border to handle edges
  cv::copyMakeBorder(src, src_border, new_radius, new_radius, new_radius,
                     new_radius, cv::BORDER_CONSTANT,
                     cv::Scalar(border_value, border_value, border_value));

  src_border.convertTo(src_border, CV_32FC3, 1 / 255.0);
  cv::cvtColor(src_border, src_lab, cv::COLOR_BGR2Lab);

  // Prepare the closeness kernel (spatial Gaussian)
  cv::Mat closeness_kernel(2 * new_radius + 1, 2 * new_radius + 1, CV_32F);
  for (int row = -new_radius; row <= new_radius; row++) {
    for (int col = -new_radius; col <= new_radius; col++) {
      closeness_kernel.at<float>(row + new_radius, col + new_radius) = std::exp(
          -0.5 * ((row * row + col * col) / (sigma_distance * sigma_distance)));
    }
  }

  // Apply bilateral filter
  for (int row_idx = new_radius; row_idx < src_lab.rows - new_radius;
       row_idx++) {
    for (int col_idx = new_radius; col_idx < src_lab.cols - new_radius;
         col_idx++) {
      filtered_value = 0.0f;
      weight_sum = 0.0f;

      for (int dy = -new_radius; dy <= new_radius; dy++) {
        for (int dx = -new_radius; dx <= new_radius; dx++) {
          filtered_value +=
              closeness_kernel.at<float>(dy + new_radius, dx + new_radius) *
              std::exp(
                  -0.5 *
                  ((src_lab.at<cv::Vec3f>(row_idx, col_idx)[0] -
                    src_lab.at<cv::Vec3f>(row_idx + dy, col_idx + dx)[0]) *
                   (src_lab.at<cv::Vec3f>(row_idx, col_idx)[0] -
                    src_lab.at<cv::Vec3f>(row_idx + dy, col_idx + dx)[0])) /
                  (sigma_range * sigma_range)) *
              src_lab.at<cv::Vec3f>(row_idx + dy, col_idx + dx)[0];

          weight_sum +=
              closeness_kernel.at<float>(dy + new_radius, dx + new_radius) *
              std::exp(
                  -0.5 *
                  ((src_lab.at<cv::Vec3f>(row_idx, col_idx)[0] -
                    src_lab.at<cv::Vec3f>(row_idx + dy, col_idx + dx)[0]) *
                   (src_lab.at<cv::Vec3f>(row_idx, col_idx)[0] -
                    src_lab.at<cv::Vec3f>(row_idx + dy, col_idx + dx)[0])) /
                  (sigma_range * sigma_range));
        }
      }

      // Normalize and assign to dst
      dst_lab.at<cv::Vec3f>(row_idx - new_radius, col_idx - new_radius)[0] =
          filtered_value / weight_sum;
      dst_lab.at<cv::Vec3f>(row_idx - new_radius, col_idx - new_radius)[1] =
          src_lab.at<cv::Vec3f>(row_idx, col_idx)[1];
      dst_lab.at<cv::Vec3f>(row_idx - new_radius, col_idx - new_radius)[2] =
          src_lab.at<cv::Vec3f>(row_idx, col_idx)[2];
    }
  }

  // Convert back to BGR color space and set the output
  cv::cvtColor(dst_lab, dst, cv::COLOR_Lab2BGR);
  dst.convertTo(dst, CV_8UC3, 255);

  return true;
}
}
