//
// Created by maxim on 10.05.24.
//

#pragma(once)
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct Roi {
public:
    int cols = 0;
    int rows = 0;
    int colWidth = 0;
    int rowWidth = 0;
    int xStart = 0;
     int yStart = 0;

    cv::Mat splitImageToSections(const cv::Mat &img) ;
    std::vector<std::pair<double,double>> calculateESF(const cv::Mat& roiImage);
    std::vector<std::pair<double,double>> calculateLSFfromESF(std::vector<std::pair<double, double>> ESF);
    std::vector<std::pair<double,double>> calculateMTFfromLSF(std::vector<std::pair<double,double>> LSF);
    double calculateFWHM(std::vector<std::pair<double,double>> LSF);
    double calculateFWTM(std::vector<std::pair<double,double>> LSF);
    std::vector<std::pair<double,double>> calculateMTFthreshold(std::pair<std::vector<double>,std::vector<double>> MTF);

private:
    std::vector<std::vector<double>> calculateSectionsIntense(const cv::Mat &roiImage);
    int findFirstElementBiggerMean(const std::vector<double>& rowSectionIntenses, const double& avgIntense);
    std::vector<int> findBorderIndexInRow(const std::vector<std::vector<double>> &sectionIntense);
    double calculatePlaneCos(const std::vector<int> &borderIndex, const std::vector<std::vector<double>> &sectionIntense) ;
    std::vector<std::vector<double>> calculateESFabscisses(const std::vector<int> &borderIndex, const double &cosPlane);
    double biSquare(double x);
    std::vector<std::pair<double, double>> calculateWindow(const std::vector<std::pair<double, double>>& ESF, double x, double window);
    double calculateSmoothingKernel(const std::vector<std::pair<double, double>>& point, double x, double window);
    std::vector<std::pair<double, double>> linearSmoothing(const std::vector<std::pair<double, double>>& ESF, double step, double window);
};

std::pair<std::vector<double>, std::vector<double>> vpTopv(const std::vector<std::pair<double,double>>& v);
cv::Mat loadImage(std::string relative_path);