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
    Segment(std::vector<cv::Point> V, cv::Mat& img) : points_(V){
        contour_ = get_contour(img);        
        circle_moment_ = calc_circle_moment();
        w0_ = W0();
    }
    
    std::vector<cv::Point> points_;
    std::vector<cv::Point> contour_;
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

    std::pair<cv::Point, double> create_min_circle() {
        cv::Point center;

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
        return std::make_pair<cv::Point, double>(cv::Point(center.y, center.x), max_distance/2);
    }

    double calc_circle_moment() {
        std::pair<cv::Point, double> circle = create_min_circle();
        return points_.size() / (M_PI * pow(circle.second,2));
    }

    std::vector<cv::Point> get_contour(cv::Mat& img) {
        std::vector<cv::Point> V;
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
            if (V.size() > 100) {
                Segment s(V, img);
                S.push_back(s);
            }
        }
    } 
    return S;
}

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

void morphology_operation(cv::Mat& img, int kernel_size, int set_value, int default_value) {
    cv::Mat tmp = copy_image(img);

    for (int i = kernel_size - 2; i < img.rows - kernel_size + 2; ++i) {
        for (int j = kernel_size - 2; j < img.cols - kernel_size + 2; ++j) {
            int new_value = default_value;
            for (int x = 0; x < kernel_size; ++x) {
                for (int y = 0; y < kernel_size; ++y) {
                    if (tmp.at<uchar>(i - (kernel_size - 2) + x, j - (kernel_size - 2) + y) == set_value) {
                        new_value = set_value;
                    }
                }
            }
            img.at<uchar>(i,j) = new_value;
        }
    }
}

void dilate(cv::Mat& img, int kernel_size) {
    morphology_operation(img, kernel_size, 255, 0);
}

void erode(cv::Mat& img, int kernel_size) {
    morphology_operation(img, kernel_size, 0, 255);
}

void draw_segment(Segment& s, cv::Mat& img) {
    int max_y = s.get_max_y();
    int min_y = s.get_min_y();
    int max_x = s.get_max_x();
    int min_x = s.get_min_x();


    for (int i = min_x; i <= max_x; ++i) {
        img.at<cv::Vec3b>(i,max_y)[0] = 0;    
        img.at<cv::Vec3b>(i,max_y)[1] = 255;    
        img.at<cv::Vec3b>(i,max_y)[2] = 0;    

        img.at<cv::Vec3b>(i, min_y)[0] = 0;    
        img.at<cv::Vec3b>(i, min_y)[1] = 255;    
        img.at<cv::Vec3b>(i, min_y)[2] = 0;    
    }
    for (int i = min_y; i <= max_y; ++i) {

        img.at<cv::Vec3b>(min_x,i)[0] = 0;    
        img.at<cv::Vec3b>(min_x,i)[1] = 255;    
        img.at<cv::Vec3b>(min_x,i)[2] = 0;    

        img.at<cv::Vec3b>(max_x, i)[0] = 0;    
        img.at<cv::Vec3b>(max_x, i)[1] = 255;    
        img.at<cv::Vec3b>(max_x, i)[2] = 0;    
    }
    
}

bool identify_condition(Segment& s) {
    if (
        s.circle_moment_ > 0.37 &&
        s.circle_moment_ < 0.47 &&
        s.w0_ > 0.18 &&
        s.w0_ < 0.74 
    ) return true;

    return false;
}

// 0.18, 0.74
void identify_segment(cv::Mat& img, cv::Mat& original_image, Segment& s) {
    if (s.points_.size() > 500 && s.points_.size() < 5000) {
        if (s.circle_moment_ > 0.37 && s.circle_moment_ < 0.47 ) {
            draw_segment(s, original_image);
        }
    }
}

void process_image(cv::Mat& original_image) {
    cv::Mat img = convert_BGR2GRAY(original_image);
    threshold_image(img, 210);
    dilate(img, 5);
    erode(img, 3);
    std::vector<Segment> segments = seg_image(img);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

    for (auto s : segments) {
        identify_segment(img, original_image, s);
    }
}

void run_for_all_images() {
    std::vector<std::string> files {
        "./images/pobr-input2-a.jpg",
        "./images/pobr-input5-a.jpg",
        "./images/pobr-input7-a.jpg",
        "./images/pobr-input8-a.jpg"
    };

    for (auto file : files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);
        if(img.empty()) {
            std::cout << "Could not read the image: " << file << std::endl;
            return;
        }
        
        process_image(img);
        cv::imshow(file, img);
    }
}

void run_for_test() {
    std::string image_path = cv::samples::findFile("./images/pobr-input8-a.jpg");
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if(img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return;
    }
    
    process_image(img);
    cv::imshow("result", img);
} 

int main() {
    //run_for_all_images();
    run_for_all_images();
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}
