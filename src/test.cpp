#include "opencv2/opencv.hpp"
#include "opencv4/opencv2/core.hpp"
#include "../inc/OpenCVLearning.h"
#include "../inc/OpenCV.h"
#include "../inc/main.h"
// #define use_cv
// include 必须要写到.hpp 写到文件夹会报错
using namespace std;
using namespace cv;

void camera()
{
    VideoCapture capture(0);
    Mat frame;
    while (1)
    {
        capture.read(frame);
        Mat grayImage;
        cvtColor(frame, grayImage, COLOR_RGB2GRAY); // 转换为灰度图像
        Mat descriptors;
        vector<KeyPoint> keypoints;
        // 创建SIFT检测器
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        // 使用SIFT检测角点并计算描述符
        sift->detectAndCompute(grayImage, Mat(), keypoints, descriptors);
        // 绘制角点
        Mat keypoints_image;
        drawKeypoints(grayImage, keypoints, keypoints_image, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("camera", keypoints_image);
        cv::waitKey(1);
    }
}

#ifndef use_cv
int show()
{
    // BasicCV base; // 类
    const char *imagename = "/home/dzx/图片/picture.jpg"; // 此处为你自己的图片路径
    // 从文件中读入图像
    Mat img = imread(imagename, 1);
    // base.inrange(img); // 类
    // 如果读入图像失败
    if (img.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename);
        return -1;
    }
    // 显示图像
    cv::imshow("image", img);
    cv::waitKey(0);
    return 0;
}

void gray(const Mat &img)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (img.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    Mat grayImage;
    cvtColor(img, grayImage, COLOR_RGB2GRAY); // 转换为灰度图像
    cv::imshow("Grayscale Image", grayImage);
    cv::waitKey(0);
}

void hsv(const Mat &img)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (img.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV); // 转换成色彩鲜明的hsv图像
    cv::imshow("hsv", hsv);
    cv::waitKey(0);
}

void gaussianBlurImage(const Mat &img)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (img.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    // 高斯滤波器
    Mat gaussianBlurImage;
    cv::GaussianBlur(img, gaussianBlurImage, Size(5, 5), 1.5); // 5x5 高斯核，标准差1.5
    cv::imshow("Gaussian Blurred Image", gaussianBlurImage);
    cv::waitKey(0);
}

void medianBlurImage(const Mat &img)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (img.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    // 中值滤波器
    Mat medianBlurImage;
    cv::medianBlur(img, medianBlurImage, 5); // 5x5 中值核
    cv::imshow("Median Blurred Image", medianBlurImage);
    cv::waitKey(0);
}

void blurImage(const Mat &img)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (img.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    // 均值滤波器
    Mat blurImage;
    cv::blur(img, blurImage, Size(5, 5)); // 5x5 均值核
    cv::imshow("junzhi Blurred Image", blurImage);
    cv::waitKey(0);
}

void out1(const Mat &img)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (img.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    Mat element1 = getStructuringElement(MORPH_RECT, Size(15, 15)); // 获取自定义核
    Mat out1;
    cv::dilate(img, out1, element1); // 进行膨胀操作
    cv::imshow("【效果图】膨胀操作", out1);
    cv::waitKey(0);
}

void out2(const Mat &img)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (img.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    Mat element2 = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat out2;
    cv::erode(img, out2, element2); // 进行腐蚀操作
    cv::imshow("【效果图】腐蚀操作", out2);
    cv::waitKey(0);
}

void canny(const cv::Mat &grayImage)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (grayImage.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    // canny边缘检测
    Mat edges;
    double threshold1 = 50;  // 较低的阈值
    double threshold2 = 150; // 较高的阈值
    Canny(grayImage, edges, threshold1, threshold2);
    // 显示边缘检测结果
    cv::imshow("Canny Edges", edges);
    cv::waitKey(0);
}

void Harris(const cv::Mat &grayImage)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (grayImage.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    // Harris角点检测
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst1, dst1_norm, harris_image;
    // 使用Harris算法计算角点
    cv::cornerHarris(grayImage, dst1, blockSize, apertureSize, k);
    // 归一化角点响应
    cv::normalize(dst1, dst1_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    cv::convertScaleAbs(dst1_norm, harris_image);
    // 显示Harris角点检测结果
    cv::imshow("Harris Corners", harris_image);
    cv::waitKey(0);
}

void SIFT(const cv::Mat &grayImage)
{ // 使用const引用避免复制大图像数据
    // 确保img不为空
    if (grayImage.empty())
    {
        cerr << "Error: Image is empty." << endl;
        return;
    }
    Mat descriptors;
    vector<KeyPoint> keypoints;
    // 创建SIFT检测器
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    // 使用SIFT检测角点并计算描述符
    sift->detectAndCompute(grayImage, Mat(), keypoints, descriptors);
    // 绘制角点
    Mat keypoints_image;
    drawKeypoints(grayImage, keypoints, keypoints_image, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // 显示SIFT角点
    cv::imshow("SIFT Keypoints", keypoints_image);
    cv::waitKey(0);
}

void BasicCV::inrange(Mat &img)
{
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV); // 转换成色彩鲜明的hsv图像
    cv::imshow("hsv", hsv);
    cv::waitKey(0);
    imwrite("/home/dzx/图片/redback.jpg", hsv);
}
// void test()
// {
//     BasicCV base;
//     BasicCV *Base=new BasicCV;
//     cv::Mat image;
//     base.inrange(image);
//     Base->inrange(image);
// }
#endif
