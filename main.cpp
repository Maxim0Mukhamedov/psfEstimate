#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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

std::vector<std::vector<double>> calculateESFordinates(const std::vector<int> &borderIndex, const double &cosPlane) {
    std::vector<std::vector<double>> esfPointOrdinate(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            esfPointOrdinate[i][j] = (j - borderIndex[i]) * cosPlane;
        }
    }
    return esfPointOrdinate;
}

std::vector<std::pair<double, double>> calculateESFderivites(const std::vector<std::vector<double>>& X, std::vector<std::vector<double>> Y) {
    std::vector<std::pair<double, double>> LSF;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            LSF.push_back({X[i][j],Y[i][j]});
        }
    }
    std::sort(LSF.begin(),LSF.end());
    std::vector<std::pair<double,double>> trueLSF;
    double cnt = 0;
    double sum = 0;
    for (int i = 0; i < cols * rows - 1; i ++) {
       if (LSF[i + 1].first == LSF[i].first) {
           if (cnt == 0) {
               cnt = 2;
               sum = LSF[i].second;
           } else {
               cnt++;
           }
           sum += LSF[i + 1].second;
       } else {
           trueLSF.push_back({LSF[i].first,sum/cnt});
           cnt = 0;
           sum = 0;
       }
    }
    for (int i = 0; i < trueLSF.size() - 1; i ++) {
        trueLSF[i] = {trueLSF[i].first, (trueLSF[i + 1].second - trueLSF[i].second)/(trueLSF[i+1].first - trueLSF[i].first)};
    }
    trueLSF[trueLSF.size() - 1] = {trueLSF[trueLSF.size() - 1].first, trueLSF[trueLSF.size() - 2].second};

    return trueLSF;
}


int main() {

    cv::Mat img = loadImage("/home/maxim/CLionProjects/psfEstimate/testData/synthEdgeImage.png");
    cv::Mat roiImage = splitImageToSections(img);
    std::vector<std::vector<double>> sectionIntense = calculateSectionsIntense(roiImage);
    std::vector<int> borderIndex = findSectionBorderIndex(sectionIntense);
    double cosPlane = calculatePlaneCos(borderIndex);
    std::vector<std::vector<double>> esfPointOrdinate = calculateESFordinates(borderIndex, cosPlane);
    std::vector<std::pair<double, double>> LSF = calculateESFderivites(esfPointOrdinate,sectionIntense);
    for (int i = 0; i < LSF.size(); i++) {
        std::cout << LSF[i].first << ' ' << LSF[i].second << std::endl;
    }
    return 0;
}
