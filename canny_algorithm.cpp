#include "opencv2/highgui.hpp"     // to use cv::waitKey()
#include "opencv2/imgproc.hpp"     // to use cv::GaussianBlur()
#include <assert.h>                // to use assert()
#include <iostream>
#include <unistd.h>

#define name_var(x) #x;

// * defind degree threshold
// @ ====================================================
bool __45_degree(const float& Angle) {
    return ((Angle > 0) && (Angle <= 45)) || ((Angle > 180) && (Angle <= 225));
}

bool __90_degree(const float& Angle) {
    return ((Angle > 45) && (Angle <= 90)) || ((Angle > 225) && (Angle <= 270));
}

bool __135_degree(const float& Angle) {
    return ((Angle > 90) && (Angle <= 135)) || ((Angle > 270) && (Angle <= 315));
}

bool __180_degree(const float& Angle) {
    return (Angle == 0) || ((Angle > 135) && (Angle <= 180)) || ((Angle > 315) && (Angle <= 360));
}
// @ ====================================================

void Gradient_image(const cv::Mat& image_source,
                    cv::Mat& image_output,      // an empty matrix to store result
                    cv::Mat& image_Gx,          // an empty matrix to store gradient X direction value
                    cv::Mat& image_Gy,          // an empty matrix to store gradient Y direction value
                    cv::Mat_<float>& angle)     // an empty matrix to store arctan(Gy / Gx)
{
    image_Gx = cv::Mat::zeros(image_source.size(), CV_16SC1);
    image_Gy = cv::Mat::zeros(image_source.size(), CV_16SC1);
    angle = cv::Mat_<float>::zeros(image_source.size());
    image_output = cv::Mat::zeros(image_source.size(), CV_8UC1);
    int row_minus_1 = image_source.rows - 1;
    int col_minus_1 = image_source.cols - 1;

    int row = image_source.rows;
    int col = image_source.cols;

    auto point = image_source.data;
    int step = image_source.step;

    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            uchar pixel_00 = point[(i - 1) * step + j - 1];
            uchar pixel_01 = point[(i - 1) * step + j];
            uchar pixel_02 = point[(i - 1) * step + j + 1];
            uchar pixel_10 = point[i * step + j - 1];
            uchar pixel_11 = point[i * step + j];
            uchar pixel_12 = point[i * step + j + 1];
            uchar pixel_20 = point[(i + 1) * step + j - 1];
            uchar pixel_21 = point[(i + 1) * step + j];
            uchar pixel_22 = point[(i + 1) * step + j + 1];

            // float grad_x = (-1 * pixel_00) + (-2 * pixel_10) + (-1 * pixel_20) + (1 * pixel_02) + (2 * pixel_12) + (1 * pixel_22);
            int grad_x = pixel_02 + (2 * pixel_12) + pixel_22 - pixel_00 - (2 * pixel_10) - pixel_20;

            // float grad_y = (1 * pixel_00) + (2 * pixel_01) + (1 * pixel_02) + (-1 * pixel_20) + (-2 * pixel_21) + (-1 * pixel_22);
            int grad_y = pixel_00 + (2 * pixel_01) + pixel_02 - pixel_20 - (2 * pixel_21) - pixel_22;

            image_Gx.at<char16_t>(i, j) = grad_x;
            image_Gy.at<char16_t>(i, j) = grad_y;

            angle.at<float>(i, j) = atan(grad_y / ((grad_x == 0) ? 0.000001 : float(grad_x)));
            image_output.at<uchar>(i, j) = sqrt(grad_x * grad_x + grad_y * grad_y);
        }
    }
}

void non_maximum_suppression(cv::Mat& image_grad,              // image has been gradiented first
                             const cv::Mat_<float>& angle)     // image which store angel
{
    int row_minus_1 = image_grad.rows - 1;
    int col_minus_1 = image_grad.cols - 1;

    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            float Angle = angle.at<float>(i, j);
            uchar previous, next;

            if (__45_degree(Angle)) {
                previous = image_grad.at<uchar>(i - 1, j + 1);     // pixel_02
                next = image_grad.at<uchar>(i + 1, j - 1);         // pixel_20
            } else if (__90_degree(Angle)) {
                previous = image_grad.at<uchar>(i - 1, j);     // pixel_01
                next = image_grad.at<uchar>(i + 1, j);         // pixel_21
            } else if (__135_degree(Angle)) {
                previous = image_grad.at<uchar>(i - 1, j - 1);     // pixel_00
                next = image_grad.at<uchar>(i + 1, j + 1);         // pixel_22
            } else if (__180_degree(Angle)) {
                previous = image_grad.at<uchar>(i, j - 1);     // pixel_10
                next = image_grad.at<uchar>(i, j + 1);         // pixel_12
            }

            if (image_grad.at<uchar>(i, j) < previous || image_grad.at<uchar>(i, j) < next)
                image_grad.at<uchar>(i, j) = 0;
        }
    }
}

void double_threshold(cv::Mat& image_grad, const int& low, const int& high) {
    //assert(low > 0 && high);     // if low < 0, exit this function immediately
    int row_minus_1 = image_grad.rows - 1;
    int col_minus_1 = image_grad.cols - 1;
    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            uchar& value = image_grad.at<uchar>(i, j);
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
                        if (image_grad.at<uchar>(i + m, j + n) > high) {
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

    // for (int i = 1; i < row_minus_1; ++i) {
    //     for (int j = 0; j < col_minus_1; ++j) {
    //         uchar& value = image_grad.at<uchar>(i, j);
    //         bool changed = false;
    //         if (value != 255 && value != 0) {
    //             for (int m = -1; m <= 1; ++m) {
    //                 for (int n = -1; n <= 1; ++n) {
    //                     if (m == 0 && n == 0)
    //                         continue;
    //                     if (image_grad.at<uchar>(i + m, j + n) > high) {
    //                         value = 255;
    //                         changed = true;
    //                         break;
    //                     }
    //                 }
    //                 if (changed)
    //                     break;
    //             }
    //             if (!changed)
    //                 value = 0;
    //         }
    //     }
    // }
}

void Canny(const cv::Mat& image_source, cv::Mat& image_output, const int& low_threshold, const int& high_threshold) {
    assert(low_threshold <= high_threshold);
    cv::Mat image_Gx, image_Gy;
    cv::Mat_<float> angle;
    Gradient_image(image_source, image_output, image_Gx, image_Gy, angle);
    non_maximum_suppression(image_output, angle);
    double_threshold(image_output, low_threshold, high_threshold);
}

int main() {
    std::string image_path = get_current_dir_name();
    image_path.append("/Resources/final fantasy 7 remake.png");
    cv::Mat image_source = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat image_source_gray = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat image_blur, image_output;
    cv::Mat_<float> angle;

    // reduce noise
    // when you debug, you must assign "image_path" by THE FULL IMAGE PATH that you want to test.
    if (!image_source_gray.empty())
        cv::GaussianBlur(image_source_gray, image_blur, cv::Size(5, 5), 150);

    // modifing "low" and "high" to get the appropriate threshold
    int low = 40;
    int high = 60;

    Canny(image_blur, image_output, low, high);

    std::string name_image = name_var(image_output);

    cv::imwrite(name_image.append(".png"), image_output);
    // cv::namedWindow("image source", cv::WINDOW_NORMAL);
    // cv::imshow("image source", image_source);

    cv::namedWindow(name_image, cv::WINDOW_NORMAL);
    cv::imshow(name_image, image_output);
    cv::waitKey(0);
    return 0;
}