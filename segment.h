#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <cmath>
#include <set>
#include <vector>
#include "point.h"


class Segment {
public:
    Segment(std::vector<Point> V, cv::Mat& img) : points_(V){
        contour_ = get_contour(img);        
        circle_moment_ = calc_circle_moment();
        w0_ = W0();
    }
    
    std::vector<Point> points_;
    std::vector<Point> contour_;
    double circle_moment_;
    double w0_;

    int area() {
        return points_.size();
    }

    int get_min_x() {
        int minValue = contour_[0].x;
        for(auto p : contour_) {
            if (p.x < minValue) {
                minValue = p.x;
            }
        }
        return minValue;
    }

    int get_max_x() {
        int maxValue = contour_[0].x;
        for(auto p : contour_) {
            if (p.x > maxValue) {
                maxValue = p.x;
            }
        }
        return maxValue;
    }

    int get_min_y() {
        int minValue = contour_[0].y;
        for(auto p : contour_) {
            if (p.y < minValue) {
                minValue = p.y;
            }
        }
        return minValue;
    }

    int get_max_y() {
        int maxValue = contour_[0].y;
        for(auto p : contour_) {
            if (p.y > maxValue) {
                maxValue = p.y;
            }
        }
        return maxValue;
    }

    std::pair<Point, double> create_min_circle() {
        Point center;

        double max_distance = 0;
        for (int i = 0; i < contour_.size(); ++i) {
            for (int j = 0; j < contour_.size(); ++j) {

                double distance = sqrt(pow(contour_[i].x - contour_[j].x, 2) + pow(contour_[i].y - contour_[j].y, 2));
                if (max_distance < distance) {
                    max_distance = distance;
                    center.x = (contour_[i].x + contour_[j].x)/2;
                    center.y = (contour_[i].y + contour_[j].y)/2;
                }
            }
        }
        return std::make_pair<Point, double>(Point(center.y, center.x), max_distance/2);
    }

    double calc_circle_moment() {
        std::pair<Point, double> circle = create_min_circle();
        return points_.size() / (M_PI * pow(circle.second,2));
    }

    std::vector<Point> get_contour(cv::Mat& img) {
        std::vector<Point> V;
        for (auto p : points_) {
            bool isBorder = false;
            if (p.x == 0 || p.x > 0 && img.at<uchar>(p.x-1, p.y) == 0) isBorder = true;
            if (p.y == 0 || p.y > 0 && img.at<uchar>(p.x, p.y-1) == 0) isBorder = true;
            if (p.x == img.rows - 1 || p.x < img.rows - 1 && img.at<uchar>(p.x + 1, p.y) == 0) isBorder = true;
            if (p.y == img.rows - 1 || p.y < img.cols - 1 && img.at<uchar>(p.x, p.y + 1) == 0) isBorder = true;

            if (isBorder) {
                V.push_back(p);
            }
        }
        return V;
    }

    long double W0() {
        return contour_.size() / (2*sqrt(M_PI * points_.size())) - 1.0;
    }
};
