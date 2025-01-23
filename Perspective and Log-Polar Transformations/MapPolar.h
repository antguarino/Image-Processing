/** Interface file for finding map coordinates for MapPolar
 *
 *  \file ipcv/geometric_transformation/MapPolar.h
 *  \author Anthony Guarino (ag4933@rit.edu)
 *  \date 8 Oct 2024
 */

#pragma once

#include <opencv2/core.hpp>

#include <Eigen/Dense>

namespace ipcv {

bool MapPolar(const cv::Mat src, const bool use_log, cv::Mat& map1,
              cv::Mat& map2);
}