#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <implot.h>
#include <imgui.h>

const int cols = 20;
const int rows = 20;
const int colWidth = 10;
const int rowWidth = 10;
const int xStart = 0;
const int yStart = 0;

cv::Mat loadImage(std::string relative_path) {
    std::string path = cv::samples::findFile(relative_path);
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    img.convertTo(img, CV_8UC1);
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    return img;
}

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

int findMeanElementIndex(const std::vector<double>& rowSectionIntenses, const double& avgIntense) {
    return  static_cast<int>(std::upper_bound(rowSectionIntenses.begin(), rowSectionIntenses.end(), avgIntense) -
            rowSectionIntenses.begin());
}

std::vector<int> findSectionBorderIndex(const std::vector<std::vector<double>> &sectionIntense) {
    std::vector<int> indexMeanIntense(rows);
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += static_cast<double>(sectionIntense[i][j]);
        }
        double avgIntense = sum / static_cast<double>(cols);
        indexMeanIntense[i] = findMeanElementIndex(sectionIntense[i],avgIntense);
    }
    return indexMeanIntense;
}

double calculatePlaneCos(const std::vector<int> &borderIndex) {
    return rows / std::sqrt((borderIndex[rows - 1] - borderIndex[0]) * (borderIndex[rows - 1] - borderIndex[0]) + rows * rows);
}

std::vector<std::vector<double>> calculateESFabscisses(const std::vector<int> &borderIndex, const double &cosPlane) {
    std::vector<std::vector<double>> esfPointOrdinate(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            esfPointOrdinate[i][j] = (j - borderIndex[i]) * cosPlane;
        }
    }
    return esfPointOrdinate;
}

std::vector<std::pair<double, double>> averageByAbscisse(std::vector<std::pair<double, double>> points) {
    std::sort(points.begin(),points.end());
    std::vector<std::pair<double,double>> averagePoints;
    double cnt = 0;
    double sum = 0;
    for (int i = 0; i < cols * rows - 1; i ++) {
        if (points[i + 1].first == points[i].first) {
            if (cnt == 0) {
                cnt = 2;
                sum = points[i].second;
            } else {
                cnt++;
            }
            sum += points[i + 1].second;
        } else {
            averagePoints.push_back({points[i].first,sum/cnt});
            cnt = 0;
            sum = 0;
        }
    }
    return averagePoints;
}

std::vector<std::pair<double,double>> calculateESF(const cv::Mat& roiImage) {
    std::vector<std::vector<double>> sectionIntense = calculateSectionsIntense(roiImage);
    std::vector<int> borderIndex = findSectionBorderIndex(sectionIntense);
    double cosPlane = calculatePlaneCos(borderIndex);
    std::vector<std::vector<double>> esfPointAbscisses = calculateESFabscisses(borderIndex, cosPlane);
    std::vector<std::pair<double, double>> ESF;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ESF.push_back({esfPointAbscisses[i][j],sectionIntense[i][j]});
        }
    }
    ESF = averageByAbscisse(ESF);
    return ESF;
}


std::vector<std::pair<double, double>> calculateLSF(const cv::Mat& roiImage) {
    std::vector<std::pair<double, double>> ESF = calculateESF(roiImage);
    std::vector<std::pair<double, double>>& LSF = ESF;
    for (int i = 0; i < ESF.size() - 1; i ++) {
        LSF[i] = {ESF[i].first, (ESF[i + 1].second - ESF[i].second)/(ESF[i+1].first - ESF[i].first)};
    }
    LSF[ESF.size() - 1] = {ESF[ESF.size() - 1].first, ESF[ESF.size() - 2].second};
    return LSF;
}

int main() {

    cv::Mat img = loadImage("/home/maxim/CLionProjects/psfEstimate/testData/synthEdgeImage.png");
    cv::Mat roiImage = splitImageToSections(img);
    std::vector<std::pair<double, double>> ESF = calculateESF(roiImage);
    std::vector<std::pair<double, double>> LSF = calculateLSF(roiImage);

    return 0;
}
