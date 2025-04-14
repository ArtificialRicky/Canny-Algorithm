// gradient_image.cu
#include "gradient_image.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>

// 定义角度阈值判断函数
bool is_45_degree(const float &Angle) {
    return (Angle > 0 && Angle <= 45) || (Angle > 180 && Angle <= 225);
}

bool is_90_degree(const float &Angle) {
    return (Angle > 45 && Angle <= 90) || (Angle > 225 && Angle <= 270);
}

bool is_135_degree(const float &Angle) {
    return (Angle > 90 && Angle <= 135) || (Angle > 270 && Angle <= 315);
}

bool is_180_degree(const float &Angle) {
    return (Angle == 0) || (Angle > 135 && Angle <= 180) || (Angle > 315 && Angle <= 360);
}

void Gradient_image(const cv::Mat &img_src,
    cv::Mat &img_out,      // an empty matrix to store result
    cv::Mat_<float> &angle)     // an empty matrix to store arctan(Gy / Gx)
{
    angle = cv::Mat_<float>::zeros(img_src.size());
    img_out = cv::Mat::zeros(img_src.size(), CV_8UC1);
    int row_minus_1 = img_src.rows - 1;
    int col_minus_1 = img_src.cols - 1;

    // int row = img_src.rows;
    // int col = img_src.cols;

    auto point = img_src.data;
    int step = img_src.step;

    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            uchar pixel_00 = point[(i - 1) * step + j - 1];
            uchar pixel_01 = point[(i - 1) * step + j];
            uchar pixel_02 = point[(i - 1) * step + j + 1];
            uchar pixel_10 = point[i * step + j - 1];
            // uchar pixel_11 = point[i * step + j];
            uchar pixel_12 = point[i * step + j + 1];
            uchar pixel_20 = point[(i + 1) * step + j - 1];
            uchar pixel_21 = point[(i + 1) * step + j];
            uchar pixel_22 = point[(i + 1) * step + j + 1];

            // float grad_x = (-1 * pixel_00) + (-2 * pixel_10) + (-1 * pixel_20) + (1 * pixel_02) + (2 * pixel_12) + (1 * pixel_22);
            float grad_x = pixel_02 + (2 * pixel_12) + pixel_22 - pixel_00 - (2 * pixel_10) - pixel_20;

            // float grad_y = (1 * pixel_00) + (2 * pixel_01) + (1 * pixel_02) + (-1 * pixel_20) + (-2 * pixel_21) + (-1 * pixel_22);
            float grad_y = pixel_00 + (2 * pixel_01) + pixel_02 - pixel_20 - (2 * pixel_21) - pixel_22;

            angle.at<float>(i, j) = atan(grad_y / (grad_x == 0 ? 0.00001 : grad_x));
            img_out.at<uchar>(i, j) = sqrt(grad_x * grad_x + grad_y * grad_y);
        }
    }
}

// CUDA 核函数：每个线程处理图像中一个非边界像素
__global__ void GradientImageKernel(const unsigned char* src, 
                                      unsigned char* dst, 
                                      float* angle, 
                                      int rows, int cols, 
                                      int step)
{
    // 计算线程对应的 (i, j) 像素位置
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 忽略边界像素，避免越界
    if (i < 1 || i >= rows - 1 || j < 1 || j >= cols - 1)
        return;
    
    int index = i * step + j;

    // 读取 3×3 邻域内的像素
    unsigned char pixel_00 = src[(i - 1) * step + (j - 1)];
    unsigned char pixel_01 = src[(i - 1) * step + j];
    unsigned char pixel_02 = src[(i - 1) * step + (j + 1)];
    unsigned char pixel_10 = src[i * step + (j - 1)];
    unsigned char pixel_12 = src[i * step + (j + 1)];
    unsigned char pixel_20 = src[(i + 1) * step + (j - 1)];
    unsigned char pixel_21 = src[(i + 1) * step + j];
    unsigned char pixel_22 = src[(i + 1) * step + (j + 1)];

    // 使用 Sobel 算子计算水平和垂直梯度
    float grad_x = pixel_02 + (2 * pixel_12) + pixel_22 - pixel_00 - (2 * pixel_10) - pixel_20;
    float grad_y = pixel_00 + (2 * pixel_01) + pixel_02 - pixel_20 - (2 * pixel_21) - pixel_22;
    
    // 计算梯度幅值
    float grad = sqrtf(grad_x * grad_x + grad_y * grad_y);
    dst[index] = (unsigned char)grad;
    
    // 计算梯度方向，使用 atan2f，更稳健；当 grad_x == 0 时，用极小值代替
    angle[i * cols + j] = atan2f(grad_y, (grad_x == 0 ? 0.00001f : grad_x));
}

// 主机端 CUDA 包装函数：调用核函数计算梯度
void Gradient_image_cuda(const cv::Mat &img_src,
                         cv::Mat &img_out,
                         cv::Mat_<float> &angle)
{
    int rows = img_src.rows;
    int cols = img_src.cols;
    size_t step = img_src.step; // 每行字节数

    // 初始化输出图像和角度矩阵
    img_out.create(rows, cols, CV_8UC1);
    angle.create(rows, cols);

    // 申请设备内存
    unsigned char *d_src = nullptr, *d_dst = nullptr;
    float *d_angle = nullptr;
    size_t img_size = rows * step;

    cudaMalloc((void**)&d_src, img_size);
    cudaMalloc((void**)&d_dst, img_size);
    cudaMalloc((void**)&d_angle, rows * cols * sizeof(float));

    // 将输入图像数据复制到设备内存
    cudaMemcpy(d_src, img_src.data, img_size, cudaMemcpyHostToDevice);

    // 定义线程块和网格尺寸
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // 启动 CUDA 核函数
    GradientImageKernel<<<grid, block>>>(d_src, d_dst, d_angle, rows, cols, step);
    cudaDeviceSynchronize();

    // 将结果复制回主机内存
    cudaMemcpy(img_out.data, d_dst, img_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(angle.data, d_angle, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_angle);
}

// 非极大值抑制：仅保留局部极大值
void non_maximum_suppression(cv::Mat &img_out, const cv::Mat_<float> &angle)
{
    int row_minus_1 = img_out.rows - 1;
    int col_minus_1 = img_out.cols - 1;

    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            float Angle = angle.at<float>(i, j);
            uchar &value = img_out.at<uchar>(i, j);
            uchar previous, next;

            if (is_45_degree(Angle)) {
                previous = img_out.at<uchar>(i - 1, j + 1);
                next = img_out.at<uchar>(i + 1, j - 1);
            } else if (is_90_degree(Angle)) {
                previous = img_out.at<uchar>(i - 1, j);
                next = img_out.at<uchar>(i + 1, j);
            } else if (is_135_degree(Angle)) {
                previous = img_out.at<uchar>(i - 1, j - 1);
                next = img_out.at<uchar>(i + 1, j + 1);
            } else if (is_180_degree(Angle)) {
                previous = img_out.at<uchar>(i, j - 1);
                next = img_out.at<uchar>(i, j + 1);
            }

            if (value < previous || value < next)
                value = 0;
        }
    }
}

// 双阈值处理：根据低/高阈值确定边缘像素
void double_threshold(cv::Mat &img_out, const int &low, const int &high) 
{
    assert(low >= 0 && high >= 0 && low <= high);
    
    int row_minus_1 = img_out.rows - 1;
    int col_minus_1 = img_out.cols - 1;
    
    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            uchar &value = img_out.at<uchar>(i, j);
            bool changed = false;
            if (value < low)
                value = 0;
            else if (value > high)
                value = 255;
            else {
                for (int m = -1; m <= 1; ++m) {
                    for (int n = -1; n <= 1; ++n) {
                        if (m == 0 && n == 0)
                            continue;
                        if (img_out.at<uchar>(i + m, j + n) > high) {
                            value = 255;
                            changed = true;
                            break;
                        }
                    }
                    if (changed)
                        break;
                }
                if (!changed)
                    value = 0;
            }
        }
    }
}

// Canny 边缘检测：调用 CUDA 计算梯度，然后进行非极大值抑制和双阈值处理
void Canny(const cv::Mat &img_src, cv::Mat &img_out,
           const int &low_threshold, const int &high_threshold) 
{
    assert(low_threshold <= high_threshold);
    cv::Mat_<float> angle;

    int64 t1 = cv::getTickCount();
    Gradient_image_cuda(img_src, img_out, angle);
    // Gradient_image(img_src, img_out, angle);
    int64 t2 = cv::getTickCount();
    std::cout << "Gradient_image_cuda: " << (t2 - t1) / cv::getTickFrequency() << " sec\n";

    non_maximum_suppression(img_out, angle);
    int64 t3 = cv::getTickCount();
    std::cout << "non_maximum_suppression: " << (t3 - t2) / cv::getTickFrequency() << " sec\n";

    double_threshold(img_out, low_threshold, high_threshold);
    int64 t4 = cv::getTickCount();
    std::cout << "double_threshold: " << (t4 - t3) / cv::getTickFrequency() << " sec\n";

    std::cout << "Total Canny time: " << (t4 - t1) / cv::getTickFrequency() << " sec\n";
}