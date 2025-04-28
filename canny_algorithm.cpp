// canny_algorithm.cpp
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "canny.h"

int main() {
    cv::Mat img_src = cv::imread("input/pic_4k.png", cv::IMREAD_COLOR);
    if (img_src.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    std::cout << "Image size: " << img_src.cols << " x " << img_src.rows << std::endl;

    int64 start = cv::getTickCount();
    
    cv::Mat img_gray;
    cv::cvtColor(img_src, img_gray, cv::COLOR_BGR2GRAY);
    cv::Mat img_blur;
    if (!img_gray.empty())
        cv::GaussianBlur(img_gray, img_blur, cv::Size(5, 5), 150);

    int low = 40;
    int high = 60;

    cv::Mat img_out;
    Canny(img_blur, img_out, low, high);
    cv::imwrite("output/canny/output.png", img_out);

    int64 end = cv::getTickCount();
    std::cout << "==> Total runtime: " << (end - start) / cv::getTickFrequency() << " sec\n";

    // cv::imshow("Canny Edge Detection", img_out);
    // cv::waitKey(0);
    return 0;
}