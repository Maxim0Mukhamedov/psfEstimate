//
// Created by maxim on 29.03.24.
//
#include "iostream"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <psfc.hpp>

cv::Mat createTestImage(const int& width, const int& height, double angle) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Scalar edgeColor(0,0,0);
    angle = angle/180.0f * CV_PI;
    double tgAngle = std::tan(angle);
    int bottomSide = std::ceil(tgAngle * height);
    std::vector<cv::Point> points{cv::Point(0,0),cv::Point(width/2,0),cv::Point(width/2 - bottomSide,height),cv::Point(0,height)};
    cv::fillPoly(img,points, edgeColor);
    return img;
}

int main() {
    cv::Mat img = createTestImage(500,500,-20);
    cv::GaussianBlur(img,img,cv::Size(5,5),10);
    cv::imwrite("/home/maxim/CLionProjects/psfEstimate/testData/psfTestImage.png",img);
    cv::imshow("PSF Test Image", img);
    cv::waitKey(0);
    return 0;
}

