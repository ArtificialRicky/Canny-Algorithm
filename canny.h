#ifndef GRADIENT_IMAGE_H
#define GRADIENT_IMAGE_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void Gradient_image(const cv::Mat &img_src,
                    cv::Mat &img_out,      // an empty matrix to store result
                    cv::Mat_<float> &angle);    // an empty matrix to store arctan(Gy / Gx)

void Gradient_image_cuda(const cv::Mat &img_src,
                         cv::Mat &img_out,
                         cv::Mat_<float> &angle);

void non_maximum_suppression(cv::Mat &img_out, const cv::Mat_<float> &angle);

void double_threshold(cv::Mat &img_out, const int &low, const int &high);

void Canny(const cv::Mat &img_src, cv::Mat &img_out,
           const int &low_threshold, const int &high_threshold);

#endif // GRADIENT_IMAGE_H