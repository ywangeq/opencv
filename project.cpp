#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include<cstdio>
#include<cstdlib>
#include<ctime>

using namespace cv;
using namespace std;
int LQIterationThreshold(Mat& src)  //自适应求梯度的算法
{
	//CV_Assert(1 == src.channels());
	int width = src.cols;
	int height = src.rows;

	float temp = 0, Threshold = 0; 
	float Ub = 0, Uo = 0;//Ub是背景的平均值，Uo是目标的平均值 
	int Number_black = 4, Number_object = 0;//nUb是背景区域像素的数目nUb是背景区域像素初始为4，nUo是目标区域像素的数目  
	Number_object = width * height - Number_black;
	Ub = (float)(src.at<float>(height / 2, width / 2) + src.at<float>(height / 2, width / 2 - 1) + src.at<float>(height / 2 - 1, 0) +
		src.at<float>(height / 2 - 1, width / 2 - 1)) / Number_black;

	Uo = Uo - Ub * Number_black;
	Uo /= Number_object;
	Threshold = (Ub + Uo) / 2;
	while (temp != Threshold)
	{
		Ub = 0;
		Uo = 0;
		Number_black = 0;
		Number_object = 0;
		for (int y = 0; y<height; y++)
		{
			float* srcdata = src.ptr<float>(y);
			for (int x = 0; x<width; x++)
			{

				if ((float)srcdata[x] < Threshold)//if f(i,j)<T, the f(i,j) belongs to the backgroud region               
				{
					Ub += (float)srcdata[x];
					Number_black++;
				}
				else
				{
					Uo += (float)srcdata[x];
					Number_object++;
				}
			}
		}
		Ub /= (Number_black + 1);
		Uo /= (Number_object + 1);
		temp = Threshold;
		Threshold = (Ub + Uo) / 2;
	}
	return Threshold;
	}

void calculate_gredient_map(Mat& src, Mat& norm){
	Mat gray,x,y,dir;
	if (src.channels() == 1)
	{
		src.copyTo(gray);

	}
	else{
		cvtColor(src, gray, CV_BGR2GRAY);
	}
	Sobel(gray, x, CV_32F, 1, 0, 3);
	Sobel(gray, y, CV_32F, 0, 1, 3);
	
	
	cartToPolar(x, y, norm, dir);
}
void mapker(Mat& dx, Mat& dy, double width, double height, Mat& map){

	int i_x, i_y;
	int j_x, j_y;
	height = dx.rows;
	width = dx.cols;



	Mat map_x(height, width, 0, Scalar(0));
	Mat map_y(height, width, 0, Scalar(0));



	for (i_x = 1; i_x < height-1; i_x++)
	{
		uchar* data1 = dx.ptr<uchar>(i_x);

		for (j_x = 1; j_x < width-1; j_x++)
		{
			data1[j_x] = data1[j_x];

			if ((dx.at<uchar>(i_x, j_x)>40) && (dx.at<uchar>(i_x, j_x - 1)>30) && (dx.at<uchar>(i_x, j_x + 1)>30))
			{
				map_x.at<uchar>(i_x, j_x) = 1;
			}
			else
			{
				map_x.at<uchar>(i_x, j_x) = 0;

			}

		}

	}


	for (i_y = 1; i_y < height-1; i_y++) //i_y 行
	{
		uchar* data2 = dy.ptr<uchar>(i_y);
		for (j_y = 1; j_y < width-1; j_y++)
		{
			data2[i_y] = data2[i_y];

			if ((dy.at<uchar>(i_y, j_y)>40) && (dy.at<uchar>(i_y, j_y - 1)>30) && (dy.at<uchar>(i_y, j_y + 1)>30))
			{

				map_y.at<uchar>(i_y, j_y) = 1;


			}
			else{

				map_y.at<uchar>(i_y, j_y) = 0;

			}
		}

	}
	map = map_x + map_y;
	//cout << map << endl;

}
void adp_gaussian_sigma(Mat&src, Mat& dst, double sigm, int n, float up, float down, double width, double height){  //up 小 down 大于1

	Mat	kernel_up = getGaussianKernel(n, sigm*up, 6);
	Mat kernel_down = getGaussianKernel(n, sigm*down, 6);
	Mat kernel_nor = getGaussianKernel(n, sigm, 6);
	Mat t_up, t_down, t_nor;
	cout << kernel_down << endl;
	cout << kernel_up << endl;
	t_up = kernel_up.t();
	t_down = kernel_down.t();
	t_nor = kernel_nor.t();
	cout << t_up << endl;
	cout << t_down << endl;
	if (src.channels() != 1)
		cvtColor(src, dst, CV_RGB2GRAY);

	else
	{
		src.copyTo(dst);
	}
	Mat dx;
	Mat dy;
	Mat abs_dx;
	Mat abs_dy;





	Sobel(dst, dx, CV_64F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(dx, abs_dx);

	Sobel(dst, dy, CV_64F, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(dy, abs_dy);
	int counter = 0;
	Mat need;

	mapker(abs_dx, abs_dy, width, height, need);
	
	Mat temp(height, width, 0);
	//cout << "widht         "<<need << endl;
	//imshow("temp", dst);

	for (int i = 0; i < height; i++){ // (width,5)==(j,i)
		if (i == 0)  //i=0
		{

			for (int j = 0; j < width; j++)
			{

				if (j == 0)
				{
					//	cout << "(i,j) point  " << i << " " << j << endl;
					temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i + 1)[j + 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +  //00-00
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i + 2)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j + 2] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2];

				}
				else if (j < width - 1)
				{
					//	cout << "(i,j) edge" << i << " " << j << endl;
					temp.ptr< uchar>(i)[j] = temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i + 1)[j - 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i + 1)[j + 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2];

				}
				else if (j == width - 1)
				{
					//	cout << "(i,j) point  " << i << " " << j << endl;

					temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i + 1)[j - 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j - 2] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +  //00-00
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i + 2)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2];
				}
			}
		}


		if (i == height - 1)  //i=0
		{

			for (int j = 0; j < width; j++)
			{

				if (j == 0)
				{
					//	cout << "(i,j) point " << i << " " << j << endl;

					temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j + 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i - 2)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +  //00-00
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j + 2] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2];
				}
				else if (j < width - 1){
					//	cout << "(i,j)edge  " << i << " " << j << endl;

					temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j - 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j + 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +  //00-00
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2];
				}
				else if (j == width - 1)
				{
					//cout << "(i,j) point " << i << " " << j << endl;
					temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j - 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j - 2] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +  //00-00
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j - 2] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2];

				}
			}
		} //i=width
		if ((i > 0) && (i < height - 1)){
			for (int j = 0; j < width; j++){
				if (j == 0)
				{
					//	cout << "(i,j) edge " << i << " " << j << endl;
					temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j + 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i + 1)[j + 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2] +
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +  //00-00
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0];
				}
				else if (j == width - 1){
					temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j - 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i + 1)[j - 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +  //00-00
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
						dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1];
				}
				else if ((j > 0) && (j < width - 1))
				{
					int p = 0;//need.ptr< uchar>(i)[j];
					
					//cout << "(i,j) 内部" << i << " " << j << endl;
					if (p =2){
						temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_up.ptr<double>(1)[0] * t_up.ptr<double>(0)[1] +
							dst.ptr<uchar>(i - 1)[j - 1] * kernel_up.ptr<double>(0)[0] * t_up.ptr<double>(0)[0] +
							dst.ptr<uchar>(i)[j - 1] * kernel_up.ptr<double>(1)[0] * t_up.ptr<double>(0)[0] +
							dst.ptr<uchar>(i + 1)[j - 1] * kernel_up.ptr<double>(2)[0] * t_up.ptr<double>(0)[0] +
							dst.ptr<uchar>(i - 1)[j] * kernel_up.ptr<double>(0)[0] * t_up.ptr<double>(0)[1] +  //00-00
							dst.ptr<uchar>(i + 1)[j] * kernel_up.ptr<double>(2)[0] * t_up.ptr<double>(0)[1] +
							dst.ptr<uchar>(i - 1)[j + 1] * kernel_up.ptr<double>(0)[0] * t_up.ptr<double>(0)[2] +
							dst.ptr<uchar>(i + 1)[j + 1] * kernel_up.ptr<double>(2)[0] * t_up.ptr<double>(0)[2] +
							dst.ptr<uchar>(i)[j + 1] * kernel_up.ptr<double>(1)[0] * t_up.ptr<double>(0)[2];
						
					}
					else if (p =1){
						temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[1] +
							dst.ptr<uchar>(i - 1)[j - 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[0] +
							dst.ptr<uchar>(i)[j - 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[0] +
							dst.ptr<uchar>(i + 1)[j - 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[0] +
							dst.ptr<uchar>(i - 1)[j] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[1] +  //00-00
							dst.ptr<uchar>(i + 1)[j] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[1] +
							dst.ptr<uchar>(i - 1)[j + 1] * kernel_nor.ptr<double>(0)[0] * t_nor.ptr<double>(0)[2] +
							dst.ptr<uchar>(i + 1)[j + 1] * kernel_nor.ptr<double>(2)[0] * t_nor.ptr<double>(0)[2] +
							dst.ptr<uchar>(i)[j + 1] * kernel_nor.ptr<double>(1)[0] * t_nor.ptr<double>(0)[2];
					}
					else if (p = 0){
						temp.ptr< uchar>(i)[j] = dst.ptr<uchar>(i)[j] * kernel_down.ptr<double>(1)[0] * t_down.ptr<double>(0)[1] +
							dst.ptr<uchar>(i - 1)[j - 1] * kernel_down.ptr<double>(0)[0] * t_down.ptr<double>(0)[0] +
							dst.ptr<uchar>(i)[j - 1] * kernel_down.ptr<double>(1)[0] * t_down.ptr<double>(0)[0] +
							dst.ptr<uchar>(i + 1)[j - 1] * kernel_down.ptr<double>(2)[0] * t_down.ptr<double>(0)[0] +
							dst.ptr<uchar>(i - 1)[j] * kernel_down.ptr<double>(0)[0] * t_down.ptr<double>(0)[1] +  //00-00
							dst.ptr<uchar>(i + 1)[j] * kernel_down.ptr<double>(2)[0] * t_down.ptr<double>(0)[1] +
							dst.ptr<uchar>(i - 1)[j + 1] * kernel_down.ptr<double>(0)[0] * t_down.ptr<double>(0)[2] +
							dst.ptr<uchar>(i + 1)[j + 1] * kernel_down.ptr<double>(2)[0] * t_down.ptr<double>(0)[2] +
							dst.ptr<uchar>(i)[j + 1] * kernel_down.ptr<double>(1)[0] * t_down.ptr<double>(0)[2];
					}
				}
			}
		}

	}
	dst = temp;
}

int main(){
	Mat gray,adp_gray;
	Mat grid;
	Mat output;
	Mat image = imread("2.png");
	cvtColor(image, gray, CV_BGR2GRAY);
	bilateralFilter(gray, gray,3,9,3);
	imshow("gray", gray);
	
	adp_gaussian_sigma(gray, adp_gray,1, 3,0.5,2, gray.cols, gray.rows);
	imwrite("ada_gray.jpg",adp_gray);
	calculate_gredient_map(adp_gray, grid);
	double high = LQIterationThreshold(grid);
	double low = high / 3;
	imshow("adapt_gray", adp_gray);
	Canny(adp_gray, output, high, low, 3);
	cout << "high " << high << "   " << "low  " << low << endl;
	imshow("adapt", output);



	image = imread("2.png");
	cvtColor(image, gray, CV_BGR2GRAY);
	imwrite("gray.jpg", gray);
	GaussianBlur(gray, gray, Size(3, 3), 1, 1);
	imshow("no", gray);
	imwrite("no_ada_gray.jpg", gray);
	Canny(gray, output, 50, 160, 3);
	imshow("no-adopt", output);
	
	waitKey(0);


}