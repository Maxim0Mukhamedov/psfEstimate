#pragma once

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

class Points {
private:
    std::vector<double> x;
    std::vector<double> y;
public:
    void push_back(std::pair<double,double> point);
    std::vector<double> getAbscisses();
    std::vector<double> getOrdinates();
    int size();
};

cv::Mat loadImage(std::string relative_path);

cv::Mat splitImageToSections(const cv::Mat &img) ;

std::vector<std::vector<double>> calculateSectionsIntense(const cv::Mat &roiImage);

int findMeanElementIndex(const std::vector<double>& rowSectionIntenses, const double& avgIntense);

std::vector<int> findSectionBorderIndex(const std::vector<std::vector<double>> &sectionIntense);

double calculatePlaneCos(const std::vector<int> &borderIndex);

std::vector<std::vector<double>> calculateESFabscisses(const std::vector<int> &borderIndex, const double &cosPlane);

std::vector<std::pair<double, double>> averageByAbscisse(std::vector<std::pair<double, double>> points);

double biSquare(double x);

std::vector<std::pair<double, double>> calculateWindow(const std::vector<std::pair<double, double>>& ESF, double x, double window);

double calculateSumWeight(const std::vector<std::pair<double, double>>& point, double x, double window);

std::vector<std::pair<double, double>> linearSmoothing(const std::vector<std::pair<double, double>>& ESF, double step);

Points calculateESF(const cv::Mat& roiImage);

Points calculateLSF(const cv::Mat& roiImage);
