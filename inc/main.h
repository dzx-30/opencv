#ifndef _MAIN_H
#define _MAIN_H

#include <iostream>
#include <string.h>
#include "OpenCVLearning.h"
#include "OpenCV.h"

using namespace std;
using namespace cv;
void camera();
int show();
void gray(const Mat &img);
void hsv(const Mat &img);
void gaussianBlurImage(const Mat &img);
void medianBlurImage(const Mat &img);
void blurImage(const Mat &img);
void out1(const Mat &img);
void out2(const Mat &img);
void canny(const cv::Mat &grayImage);
void Harris(const cv::Mat &grayImage);
void SIFT(const cv::Mat &grayImage);

#endif
