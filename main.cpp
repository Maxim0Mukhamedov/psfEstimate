#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

const int cols = 5;
const int rows = 5;
const int colWidth = 20;
const int rowWidth = 20;
const int xStart = 50;
const int yStart = 80;

cv::Mat splitImageToSections(const cv::Mat &img) {
    cv::Rect roi(xStart, yStart, cols * colWidth, rows * rowWidth);
    cv::Mat roiImage = img(roi);
    return roiImage;
}

std::vector<std::vector<double>> calculateSectionsIntense(const cv::Mat &roiImage) {
    std::vector<std::vector<double>> sectionPixelIntense(rows, std::vector<double>(cols));
    for (int i = 0; i < roiImage.rows; i++) {
        for (int j = 0; j < roiImage.cols; j++) {
            sectionPixelIntense[i / rowWidth][j / colWidth] += static_cast<double>(roiImage.at<uchar>(i, j));
        }
    }
    std::vector<std::vector<double>> sectionIntense(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sectionIntense[i][j] = sectionPixelIntense[i][j] / static_cast<double>(colWidth * rowWidth);
        }
    }
    return sectionIntense;
}

std::vector<int> findBorderIndex(const std::vector<std::vector<double>> &sectionIntense) {
    std::vector<int> indexMeanIntense(rows);
    for (int i = 0; i < rows; i++) {
        int sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += static_cast<int>(sectionIntense[i][j]);
        }
        double avgIntense = sum / static_cast<double>(cols);
        indexMeanIntense[i] = static_cast<int>(std::upper_bound(sectionIntense[i].begin(), sectionIntense[i].end(), avgIntense) -
                              sectionIntense[i].begin());
    }
    return indexMeanIntense;
}

double calculatePlaneCos(const std::vector<int> &borderIndex) {
    return rows /
           std::sqrt((borderIndex[rows - 1] - borderIndex[0]) * (borderIndex[rows - 1] - borderIndex[0]) + rows * rows);
}

std::vector<std::vector<double>> calculateESFordinates(const std::vector<int> &borderIndex, const double &cosPlane) {
    std::vector<std::vector<double>> esfPointOrdinate(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            esfPointOrdinate[i][j] = (j - borderIndex[i]) * cosPlane;
        }
    }
    return esfPointOrdinate;
}

cv::Mat loadImage(std::string relative_path) {
    std::string path = cv::samples::findFile(relative_path);
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    img.convertTo(img, CV_8UC1);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    return img;
}


int main() {

    cv::Mat img = loadImage("/home/maxim/CLionProjects/psfEstimate/testData/synthEdgeImage.png");
    cv::Mat roiImage = splitImageToSections(img);
    std::vector<std::vector<double>> sectionIntense = calculateSectionsIntense(roiImage);
    std::vector<int> borderIndex = findBorderIndex(sectionIntense);
    double cosPlane = calculatePlaneCos(borderIndex);
    std::vector<std::vector<double>> esfPointOrdinate = calculateESFordinates(borderIndex, cosPlane);

    return 0;
}
