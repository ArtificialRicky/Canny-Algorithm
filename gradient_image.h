#ifndef GRADIENT_IMAGE_H
#define GRADIENT_IMAGE_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

void Gradient_image(const cv::Mat &img_src,
                    cv::Mat &img_out,      // an empty matrix to store result
                    cv::Mat_<float> &angle);    // an empty matrix to store arctan(Gy / Gx)

// CUDA 计算梯度的封装接口
void Gradient_image_cuda(const cv::Mat &img_src,
                         cv::Mat &img_out,
                         cv::Mat_<float> &angle);

// 非极大值抑制
void non_maximum_suppression(cv::Mat &img_out, const cv::Mat_<float> &angle);

// 双阈值处理
void double_threshold(cv::Mat &img_out, const int &low, const int &high);

// 基于 CUDA 的 Canny 边缘检测接口
void Canny(const cv::Mat &img_src, cv::Mat &img_out,
           const int &low_threshold, const int &high_threshold);

#endif // GRADIENT_IMAGE_H