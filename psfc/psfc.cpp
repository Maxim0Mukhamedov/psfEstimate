#include "psfc.hpp"

void Points::push_back(std::pair<double,double> point) {
    x.push_back(point.first);
    y.push_back(point.second);
}
std::vector<double> Points::getAbscisses() {
    return x;
}
std::vector<double> Points::getOrdinates() {
    return y;
}
int Points::size() {
    return x.size();
}


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
            esfPointOrdinate[i][j] = (j - borderIndex[i]) * (cosPlane + cosPlane/(rows*cols)*std::pow(-1,i) * i );
        }
    }
    return esfPointOrdinate;
}

std::vector<std::pair<double, double>> averageByAbscisse(std::vector<std::pair<double, double>> points) {
    std::sort(points.begin(),points.end());
    std::vector<std::pair<double,double>> averagePoints;
    double cnt = 1;
    double sum = points[0].second;
    for (int i = 0; i < points.size() - 1; i ++) {
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
            cnt = 1;
            sum = points[i + 1].second;
        }
    }
    return averagePoints;
}

double biSquare(double x) {
    if (-1 <= x && x <= 1) {
        return (1-x*x)*(1-x*x);
    } else {
        return 0;
    }
}

std::vector<std::pair<double, double>> calculateWindow(const std::vector<std::pair<double, double>>& ESF, double x, double window) {
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

double calculateSumWeight(const std::vector<std::pair<double, double>>& point, double x, double window) {
    double sum = 0;
    for (int i = 0; i < point.size(); i++) {
        sum += biSquare( (point[i].first - x)/window);
    }
    return sum;
}

std::vector<std::pair<double, double>> linearSmoothing(const std::vector<std::pair<double, double>>& ESF, double step) {
    double window = 2;
    std::vector<std::pair<double, double>> smoothedESF;
    for (double x = ESF[0].first; x < ESF[ESF.size() - 1].first; x += step) {
        std::vector<std::pair<double, double>> pointsInWindow = calculateWindow(ESF, x, window);
        double Wj = calculateSumWeight(pointsInWindow,x,window);
        double sum = 0;
        for (int i = 0; i < pointsInWindow.size(); i++) {
            sum += biSquare((pointsInWindow[i].first - x) / window)/Wj * pointsInWindow[i].second;
        }
        smoothedESF.push_back({x,sum});
    }
    return smoothedESF;
}

Points calculateESF(const cv::Mat& roiImage) {
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
    ESF = linearSmoothing(ESF,0.1);
    Points esf;
    for (auto i : ESF) {
        std::cout << i.first << ' ' << i.second << std::endl;
        esf.push_back(i);
    }
    return esf;
}


Points calculateLSF(const cv::Mat& roiImage) {
    Points ESF = calculateESF(roiImage);
    Points LSF;
    for (int i = 0; i < ESF.size() - 1; i ++) {
        LSF.push_back({ESF.getAbscisses()[i], (ESF.getOrdinates()[i + 1] - ESF.getOrdinates()[i])/(ESF.getAbscisses()[i+1] - ESF.getAbscisses()[i])});
    }
    LSF.push_back({LSF.getAbscisses()[ESF.size() - 2],LSF.getOrdinates()[ESF.size() - 2]});
    return LSF;
}
