#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ϵͳѧϰopencv for one week\Chapter 10\videoprocessor.h"
using namespace cv;
using namespace std;

void processFrame(Mat&img, Mat& out);
void canny(Mat& img, Mat out){
	if (img.channels() == 3)
		cvtColor(img, out, CV_BGR2GRAY);
	Canny(out, out, 100, 200);
	threshold(out, out, 128, 255, THRESH_BINARY_INV);
}
int main(){
	VideoCapture capture("04.avi");
	if (!capture.isOpened())

		return 1;
	double rate = capture.get(CV_CAP_PROP_FPS);
	bool stop(false);
	Mat frame;
	namedWindow("Extracted Frame");
	int delay = 1000 / rate;

	while (!stop){
		if (!capture.read(frame))
			break;
			imshow("Extracted Frame", frame);
			if (waitKey(delay) >= 0)
				stop = true;

		
	}
	capture.release();
	waitKey();



	VideoProcessor processor;
	processor.setInput("04.avi");
	processor.displayInput("Current frame");
	processor.displayOutput("output frame");
	processor.setDelay(1000. / processor.getFrameRate());
	processor.setFrameProcessor(canny);
	processor.run();
	waitKey();

}