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
const int rows = 40;
const int colWidth = 3;
const int rowWidth = 3;
const int xStart = 185;
const int yStart = 30;
const double window = 1.5;

namespace psfc {
    const double W = window;
}

cv::Mat loadImage(std::string relative_path);

cv::Mat splitImageToSections(const cv::Mat &img) ;

std::vector<std::vector<double>> calculateSectionsIntense(const cv::Mat &roiImage);

int findFirstElementBiggerMean(const std::vector<double>& rowSectionIntenses, const double& avgIntense);

std::vector<int> findBorderIndexInRow(const std::vector<std::vector<double>> &sectionIntense);

double calculatePlaneCos(const std::vector<int> &borderIndex);

std::vector<std::vector<double>> calculateESFabscisses(const std::vector<int> &borderIndex, const double &cosPlane);

double biSquare(double x);

std::vector<std::pair<double, double>> calculateWindow(const std::vector<std::pair<double, double>>& ESF, double x);

double calculateSmoothingKernel(const std::vector<std::pair<double, double>>& point, double x, double window);

std::vector<std::pair<double, double>> linearSmoothing(const std::vector<std::pair<double, double>>& ESF, double step);

std::vector<std::pair<double,double>> calculateESF(const cv::Mat& roiImage);

std::vector<std::pair<double,double>> calculateLSFfromESF(std::vector<std::pair<double, double>> ESF);

std::pair<std::vector<double>, std::vector<double>> vpTopv(const std::vector<std::pair<double,double>>& v);
