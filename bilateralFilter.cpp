#include "cv.h"                             //  OpenCV ÎÄ¼þÍ·
#include "highgui.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	Mat m_SrcImg;

	m_SrcImg = imread("14.jpg", -1);

	namedWindow("Ô­Í¼Ïñ", 1);
	imshow("Ô­Í¼Ïñ", m_SrcImg);

	Mat m_ResImg;
	//Ë«±ßÆ½»¬
	bilateralFilter(m_SrcImg, m_ResImg, 30, 30*2, 30/2,4);

	namedWindow("Ë«±ßÆ½»¬", 1);
	imshow("Ë«±ßÆ½»¬", m_ResImg);
	imwrite("Ë«±ßÆ½»¬.jpg", m_ResImg);
	waitKey(0);
}