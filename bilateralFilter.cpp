#include "cv.h"                             //  OpenCV �ļ�ͷ
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

	namedWindow("ԭͼ��", 1);
	imshow("ԭͼ��", m_SrcImg);

	Mat m_ResImg;
	//˫��ƽ��
	bilateralFilter(m_SrcImg, m_ResImg, 30, 30*2, 30/2,4);

	namedWindow("˫��ƽ��", 1);
	imshow("˫��ƽ��", m_ResImg);
	imwrite("˫��ƽ��.jpg", m_ResImg);
	waitKey(0);
}