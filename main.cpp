#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <cmath>
#include <set>
#include <vector>

cv::Mat copy_image(cv::Mat& img) {
    cv::Mat tmp(img.size(), img.type());
    for (int i = 0; i < tmp.rows; ++i) {
        for (int j = 0; j < tmp.cols; ++j) {
            tmp.at<uchar>(i,j) = img.at<uchar>(i,j);
        }
    }
    return tmp;
}

class Segment {
public:
    Segment(std::vector<cv::Point> V) : points_(V){}
    
    std::vector<cv::Point> points_;
    int area() {
        return points_.size();
    }

    int get_min_x() {
        int minValue = points_[0].x;
        for(auto p : points_) {
            if (p.x < minValue) {
                minValue = p.x;
            }
        }
        return minValue;
    }

    int get_max_x() {
        int maxValue = points_[0].x;
        for(auto p : points_) {
            if (p.x > maxValue) {
                maxValue = p.x;
            }
        }
        return maxValue;
    }

    int get_min_y() {
        int minValue = points_[0].y;
        for(auto p : points_) {
            if (p.y < minValue) {
                minValue = p.y;
            }
        }
        return minValue;
    }

    int get_max_y() {
        int maxValue = points_[0].y;
        for(auto p : points_) {
            if (p.y > maxValue) {
                maxValue = p.y;
            }
        }
        return maxValue;
    }

    std::pair<cv::Point, double> create_min_circle() {
        cv::Point center;

        double max_distance = 0;
        for (int i = 0; i < points_.size(); ++i) {
            for (int j = 0; j < points_.size(); ++j) {
                double distance = sqrt(pow(points_[i].x - points_[j].x, 2) + pow(points_[i].y - points_[j].y, 2));
                if (max_distance < distance) {
                    max_distance = distance;
                    center.x = abs(points_[i].x - points_[j].x) / 2;
                    center.y = abs(points_[i].y - points_[j].y) / 2;
                }
            }
        }
        return std::make_pair<cv::Point, double>(cv::Point(center.x, center.y), max_distance/2);
    }
};

void flood_fill (cv::Mat& img, std::vector<cv::Point>& V, int x, int y) {
    if (img.at<uchar>(x,y) == 0) return;

    img.at<uchar>(x,y) = 0;
    V.push_back(cv::Point(x,y));
    if (x > 0) flood_fill(img, V, x-1, y);
    if (x < img.rows - 1) flood_fill(img, V, x+1, y);
    if (y > 0) flood_fill(img, V, x, y-1);
    if (y < img.cols - 1) flood_fill(img, V, x, y+1);
}

std::vector<Segment> seg_image(cv::Mat& img) {
    cv::Mat tmp = copy_image(img);
    std::vector<Segment> S;

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            std::vector<cv::Point> V;
            flood_fill(tmp, V, i, j);
            Segment s(V);
            S.push_back(s);
        }
    } 
    return S;
}

/*
void segment_image(cv::Mat& img) {
    std::vector<std::vector<cv::Point>> C;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(img, C, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    for (auto contour : C) {
        if (contour.size() > 65 && contour.size() < 125) {
            cv::Moments M = cv::moments(contour, true);

            int cx = M.m10 / M.m00;
            int cy = M.m01 / M.m00;

            cv::circle(img, cv::Point(cx, cy), 50, cv::Scalar(0,255,0));

            auto rect = cv::boundingRect(contour);
            cv::RotatedRect rotRect = cv::minAreaRect(contour);
            auto size = rotRect.size;

            double area = cv::contourArea(contour);
            auto ratio = area/(size.height * size.width);
            std::cout << ratio << std::endl;
            if (ratio > 0.66 && ratio < 0.726) {
                cv::rectangle(img, rect, cv::Scalar(0,0,255));
            }
        }
    }
}
*/

void threshold_image(cv::Mat& img, int threshold) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            img.at<uchar>(i,j) = img.at<uchar>(i,j) >= threshold ? 255 : 0;
        }
    }
}

cv::Mat convert_BGR2GRAY(cv::Mat& img) {
    cv::Mat gray_img(img.size(), CV_8U);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            int bValue = img.at<cv::Vec3b>(i,j)[0];
            int gValue = img.at<cv::Vec3b>(i,j)[1];
            int rValue = img.at<cv::Vec3b>(i,j)[2];
            gray_img.at<uchar>(i,j) = (int)round((bValue + gValue + rValue) / 3.0);
        }
    }
    return gray_img;
}


void dilate_image(cv::Mat& img, int kernel_size) {
    cv::Mat tmp = copy_image(img);

    for (int i = kernel_size - 2; i < img.rows - kernel_size + 2; ++i) {
        for (int j = kernel_size - 2; j < img.cols - kernel_size + 2; ++j) {
            int new_value = 0;
            for (int x = 0; x < kernel_size; ++x) {
                for (int y = 0; y < kernel_size; ++y) {
                    if (tmp.at<uchar>(i - (kernel_size - 2) + x, j - (kernel_size - 2) + y) == 255) {
                        new_value = 255;
                    }
                }
            }
            img.at<uchar>(i,j) = new_value;
        }
    }
}

void process_image(cv::Mat& img_a) {
    cv::Mat img = convert_BGR2GRAY(img_a);
    //cv::cvtColor(img_a, img_a, cv::COLOR_BGR2GRAY);
    //cv::imshow("Display window", img); 
    //cv::threshold(img_a, img_a, 160, 255, cv::THRESH_BINARY);

    threshold_image(img, 210);
    cv::Mat tmp_image = copy_image(img);

    cv::dilate(tmp_image, tmp_image, cv::Mat());
    dilate_image(img, 5);

    //cv::imshow("my algorithm", img);
    //cv::imshow("opencv", tmp_image);

//    cv::erode(img, img, cv::Mat());
    std::vector<Segment> segments = seg_image(img);
//    segment_image(img);

}

int main() {
    std::string image_path = cv::samples::findFile("./images/pobr-input2-a.jpg");
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if(img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    
    process_image(img);

    int k = cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}
