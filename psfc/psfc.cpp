#include "psfc.hpp"

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
    for (int rowIndex = 0; rowIndex < roiImage.rows; rowIndex++) {
        for (int colIndex = 0; colIndex < roiImage.cols; colIndex++) {
            sectionPixelIntense[rowIndex / rowWidth][colIndex / colWidth] += static_cast<double>(roiImage.at<uchar>(rowIndex, colIndex));
        }
    }
    std::vector<std::vector<double>> sectionIntense(rows, std::vector<double>(cols));
    for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
        for (int colIndex = 0; colIndex < cols; colIndex++) {
            sectionIntense[rowIndex][colIndex] = sectionPixelIntense[rowIndex][colIndex] / static_cast<double>(colWidth * rowWidth);
        }
    }
    return sectionIntense;
}

int findFirstElementBiggerMean(const std::vector<double>& rowSectionIntenses, const double& avgIntense) {
    return  static_cast<int>(std::upper_bound(rowSectionIntenses.begin(), rowSectionIntenses.end(), avgIntense) -
                             rowSectionIntenses.begin());
}

std::vector<int> findBorderIndexInRow(const std::vector<std::vector<double>> &sectionIntense) {
    std::vector<int> indexMeanIntense(rows);
    for (int i = 0; i < rows; i++) {
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            sum += static_cast<double>(sectionIntense[i][j]);
        }
        double avgIntense = sum / static_cast<double>(cols);
        indexMeanIntense[i] = findFirstElementBiggerMean(sectionIntense[i], avgIntense);
    }
    return indexMeanIntense;
}

// FIXME: Исправить неправильный выбор вершин катетов (не во всех рядах есть граница).
double calculatePlaneCos(const std::vector<int> &borderIndex) {
    return rows / std::sqrt((borderIndex[rows - 1] - borderIndex[0]) * (borderIndex[rows - 1] - borderIndex[0]) + rows * rows);
}

std::vector<std::vector<double>> calculateESFabscisses(const std::vector<int> &borderIndex, const double &cosPlane) {
    std::vector<std::vector<double>> esfPointOrdinate(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            esfPointOrdinate[i][j] = (j - borderIndex[i]) * (cosPlane + cosPlane/(rows*cols)*std::pow(-1,i) * i );
        }
    }
    return esfPointOrdinate;
}

double biSquare(double x) {
    if (-1 <= x && x <= 1) {
        return (1-x*x)*(1-x*x);
    } else {
        return 0;
    }
}

std::vector<std::pair<double, double>> calculateWindow(const std::vector<std::pair<double, double>>& ESF, double x) {
    std::vector<std::pair<double, double>> wp;
    for (int i = 0; i < ESF.size(); i++) {
        if ( (x - window) < ESF[i].first && ESF[i].first < (x + window)) {
            wp.push_back(ESF[i]);
        } else if (ESF[i].first > x + window){
            break;
        }
    }
    return wp;
}

double calculateSmoothingKernel(const std::vector<std::pair<double, double>>& point, double x, double window) {
    double sum = 0;
    for (int i = 0; i < point.size(); i++) {
        sum += biSquare( (point[i].first - x)/window);
    }
    return sum;
}

std::vector<std::pair<double, double>> linearSmoothing(const std::vector<std::pair<double, double>>& ESF, double step) {
    std::vector<std::pair<double, double>> smoothedESF;
    for (double x = ESF[0].first; x < ESF[ESF.size() - 1].first; x += step) {
        std::vector<std::pair<double, double>> pointsInWindow = calculateWindow(ESF, x);
        double Wj = calculateSmoothingKernel(pointsInWindow, x, window);
        double sum = 0;
        for (int i = 0; i < pointsInWindow.size(); i++) {
            sum += biSquare((pointsInWindow[i].first - x) / window)/Wj * pointsInWindow[i].second;
        }
        smoothedESF.push_back({x,sum});
    }
    return smoothedESF;
}

std::vector<std::pair<double,double>> calculateESF(const cv::Mat& roiImage) {
    std::vector<std::vector<double>> sectionIntense = calculateSectionsIntense(roiImage);
    std::vector<int> borderIndex = findBorderIndexInRow(sectionIntense);
    double cosPlane = calculatePlaneCos(borderIndex);
    std::cout << std::acos(cosPlane) * 180 / CV_PI << '\n';
    std::vector<std::vector<double>> esfPointAbscisses = calculateESFabscisses(borderIndex, cosPlane);
    std::vector<std::pair<double, double>> ESF;
    int rowsInSample = rows/4;
    for (int i = 0; i < rows*cols; i ++) {
        for(int j = 0; j < rowsInSample; j++) {
            int sampleRow = std::rand() % rows;
            for(int k = 0; k < cols; k++) {
                ESF.push_back({esfPointAbscisses[sampleRow][k],sectionIntense[sampleRow][k]});
            }
        }
        std::sort(ESF.begin(), ESF.end());
        ESF = linearSmoothing(ESF, 0.1);
    }
    return ESF;
}


std::vector<std::pair<double,double>> calculateLSFfromESF(std::vector<std::pair<double, double>> ESF) {
    std::vector<std::pair<double,double>>& LSF = ESF;
    for (int i = 0; i < ESF.size() - 1; i ++) {
        LSF[i] = {ESF[i].first, (ESF[i + 1].second - ESF[i].second)/(ESF[i+1].first - ESF[i].first)};
    }
    LSF[ESF.size() - 1] = LSF[ESF.size() - 2];
    LSF = linearSmoothing(LSF,0.1);
    return LSF;
}

std::pair<std::vector<double>, std::vector<double>> vpTopv(const std::vector<std::pair<double,double>>& v) {
    std::pair<std::vector<double>, std::vector<double>> result;
    for (auto i : v) {
        result.first.push_back(i.first);
        result.second.push_back(i.second);
    }
    return result;
}
