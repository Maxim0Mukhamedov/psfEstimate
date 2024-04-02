//
// Created by maxim on 29.03.24.
//
#include "iostream"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

int main() {
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(255, 255, 255)); // Create a gray image

    cv::Scalar edgeColor(0,0,0);
    double angle = 20.0f/180.0f * CV_PI;
    double tgAngle = std::tan(angle);
    int bottomSide = std::ceil(tgAngle * 500);
    std::cout << angle << ' ' << tgAngle << ' ' << bottomSide << std::endl;
    std::vector<cv::Point> points{cv::Point(0,0),cv::Point(250,0),cv::Point(250 - bottomSide,500),cv::Point(0,500)};
    cv::fillPoly(img,points, edgeColor);
    cv::imwrite("/home/maxim/CLionProjects/psfEstimate/testData/psfTestImage.png",img);
    cv::imshow("PSF Test Image", img);
    cv::waitKey(0);

    return 0;
}
