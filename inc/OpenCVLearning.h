#ifndef _OPENCVLEARNING_H
#define _OPENCVLEARNING_H

#include "OpenCV.h"
#include <iostream>
#include <string.h>

using namespace cv;
using namespace std;

class BasicCV
{
public:
    float data;
    static int ShowImage();    // 读取和显示
    void ColorSpace(Mat &img); // 色彩空间转换
    void MatCreate(Mat &img);  // Mat对象创建（待扩充）
    void PixelVisit(Mat &img); // 图像像素读写
    void Operator(Mat &img);   // 操作数
    // void Trackingbar(Mat &img); // 滚动条
    void TrackBar(Mat &img);       // 滚动条调节亮度和对比度
    void key(Mat &img);            // 键盘响应
    void ColorStyle(Mat &img);     // OpenCV色彩空间转换
    void BitWise(Mat &img);        // 图像像素逻辑操作 与或非
    void Channels(Mat &img);       // 通道分离与合并 RGB
    void inrange(Mat &img);        // 图像色彩空间转换
    void PixelStatistic(Mat &img); // 图像像素值统计
    void Draw(Mat &img);           // 图形几何形状绘制
    void RodomDraw(Mat &img);      // 随机数 随机颜色
    void PolylineDraw(Mat &img);   // 多边形绘制与填充
private:
    int num;
};

#endif