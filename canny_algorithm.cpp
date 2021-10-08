#include "opencv2/highgui.hpp"     // to use cv::waitKey()
#include "opencv2/imgproc.hpp"     // to use cv::GaussianBlur()
#include <assert.h>                // to use assert()

//* defind degree threshold
// @ ====================================================
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
// @ ====================================================

void Gradient_image(const cv::Mat &img_src,
                    cv::Mat &img_out,      // an empty matrix to store result
                    cv::Mat_<float> &angle)     // an empty matrix to store arctan(Gy / Gx)
{
    angle = cv::Mat_<float>::zeros(img_src.size());
    img_out = cv::Mat::zeros(img_src.size(), CV_8UC1);
    int row_minus_1 = img_src.rows - 1;
    int col_minus_1 = img_src.cols - 1;

    int row = img_src.rows;
    int col = img_src.cols;

    auto point = img_src.data;
    int step = img_src.step;

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
            float grad_x = pixel_02 + (2 * pixel_12) + pixel_22 - pixel_00 - (2 * pixel_10) - pixel_20;

            // float grad_y = (1 * pixel_00) + (2 * pixel_01) + (1 * pixel_02) + (-1 * pixel_20) + (-2 * pixel_21) + (-1 * pixel_22);
            float grad_y = pixel_00 + (2 * pixel_01) + pixel_02 - pixel_20 - (2 * pixel_21) - pixel_22;

            angle.at<float>(i, j) = atan(grad_y / (grad_x == 0 ? 0.00001 : grad_x));
            img_out.at<uchar>(i, j) = sqrt(grad_x * grad_x + grad_y * grad_y);
        }
    }
}

void non_maximum_suppression(cv::Mat &img_out,            // image has been gradiented first
                             const cv::Mat_<float> &angle)     // image which store angel
{
    int row_minus_1 = img_out.rows - 1;
    int col_minus_1 = img_out.cols - 1;

    for (int i = 1; i < row_minus_1; ++i) {
        for (int j = 1; j < col_minus_1; ++j) {
            float Angle = angle.at<float>(i, j);
            uchar &value = img_out.at<uchar>(i, j);
            uchar previous, next;

            if (is_45_degree(Angle)) {
                previous = img_out.at<uchar>(i - 1, j + 1);     // pixel_02
                next = img_out.at<uchar>(i + 1, j - 1);         // pixel_20
            
            } else if (is_90_degree(Angle)) {
                previous = img_out.at<uchar>(i - 1, j);     // pixel_01
                next = img_out.at<uchar>(i + 1, j);         // pixel_21
            
            } else if (is_135_degree(Angle)) {
                previous = img_out.at<uchar>(i - 1, j - 1);     // pixel_00
                next = img_out.at<uchar>(i + 1, j + 1);         // pixel_22
            
            } else if (is_180_degree(Angle)) {
                previous = img_out.at<uchar>(i, j - 1);     // pixel_10
                next = img_out.at<uchar>(i, j + 1);         // pixel_12
            }

            if (value < previous || value < next)
                value = 0;
        }
    }
}

void double_threshold(cv::Mat &img_out,
                      const int &low,
                      const int &high) 
{
    assert(low >= 0 && high >= 0 && low <= high);     // if (low < 0) or (high < 0) or (low > high), exit this function immediately
    
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

void Canny(const cv::Mat &img_src,
           cv::Mat &img_out,
           const int &low_threshold, 
           const int &high_threshold) 
{
    assert(low_threshold <= high_threshold);
    cv::Mat_<float> angle;     // to store angle while calculating image gradient
    Gradient_image(img_src, img_out, angle);
    non_maximum_suppression(img_out, angle);
    double_threshold(img_out, low_threshold, high_threshold);
}

int main() {

    cv::Mat img_src = cv::imread("input/touka-kirisima.png", cv::IMREAD_COLOR);

    cv::Mat img_gray;
    cv::cvtColor(img_src, img_gray, cv::COLOR_BGR2GRAY);

    // reduce noise
    cv::Mat img_blur;
    if (!img_gray.empty())
        cv::GaussianBlur(img_gray, img_blur, cv::Size(5, 5), 150);

    // modifing "low" and "high" to get the appropriate threshold
    // @ ===========
    int low = 40;
    int high = 60;
    // @ ===========

    cv::Mat img_out;
    cv::Mat_<float> angle;     // to store angle while calculating image gradient
    Gradient_image(img_blur, img_out, angle);
    cv::imwrite("output/gradient/touka-kirisima-gradient.png", img_out);

    Canny(img_blur, img_out, low, high);
    cv::imwrite("output/canny/touka-kirisima-canny.png", img_out);

    // show img_src
    cv::namedWindow("image source", cv::WINDOW_NORMAL);
    cv::imshow("image source", img_src);

    // show img_out
    cv::namedWindow("image output", cv::WINDOW_NORMAL);
    cv::imshow("image output", img_out);

    cv::waitKey(0);
    return 0;
}
