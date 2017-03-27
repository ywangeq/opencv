#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "featuretracker.h"
#include "harrisDetector.h"
using namespace cv;
using namespace std;
int main(){
	Mat frame;
	Mat gray,edge;
	Mat output;
	VideoCapture capture(0);
	
	if (!capture.isOpened())
		return 1;


	  bool stop(false);
	
	  while (!stop){

		  capture >> frame;
		  cvtColor(frame, gray, CV_BGR2GRAY);
		  GaussianBlur(gray, gray, Size(3, 3), 1, 1);
		  Canny(gray, edge, 50, 130, 3);

		  vector<KeyPoint> keypoints;


		  	vector<Point2f>	corners;
			goodFeaturesToTrack(edge, corners, 500, 0.01, 10);
			approxPolyDP(corners, corners, 5, true);
		//  FastFeatureDetector gft( 50);
		 vector<Point2f>::const_iterator it = corners.begin();
		  while (it != corners.end()-1){
		  	circle(frame, *it, 3, Scalar(255, 255, 255), 1);
			 
			//  line(frame, *it, *(it+1), Scalar(255), 1);
			  cout << *it << "    " << *(it + 1) << endl;
			  ++it;
			 

		  	}
		
		//  gft.detect(edge, keypoints);
		 // drawKeypoints(frame, keypoints, frame, Scalar(222, 244, 230), DrawMatchesFlags::DRAW_OVER_OUTIMG);
		  
		  namedWindow("1");
		  imshow("1", frame);






		  if (waitKey(30) >= 0)	{
			  stop = true;
		  }
	  }
	capture.release();
	waitKey(0);
}