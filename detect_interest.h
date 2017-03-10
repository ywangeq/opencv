#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace cv;
using namespace std;
class HarrisDetector{
private:
	Mat cornerStrength;
	Mat cornerTH;
	Mat localMax;
	int neighbourhood;
	int aperture;
	double k;
	double maxStrength;
	double threshlod;
	int nonMaxSize;
	Mat kernel;
public:
	HarrisDetector() :neighbourhood(3), aperture(3), k(0.01), maxStrength(0.0), threshlod(0.01), nonMaxSize(3){
		
	};
	void setLocalMaxWindowsize(int nonMaxSize){
		this->nonMaxSize = nonMaxSize;
	};
		void detect(const Mat& image){
			cornerHarris(image, cornerStrength, neighbourhood, aperture, k);
			double minStrength;
			minMaxLoc(cornerStrength, &minStrength, &maxStrength);
			Mat dilated;
			dilate(cornerStrength, dilated, Mat());
			compare(cornerStrength, dilated, localMax, CMP_EQ);

			
	}
		Mat getCornerMap(double qualityLevel){
			Mat cornerMap;
			threshlod = qualityLevel*maxStrength;
			cv::threshold(cornerStrength, cornerTH, threshlod, 255, THRESH_BINARY);
			cornerTH.convertTo(cornerMap, CV_8U);
			bitwise_and(cornerMap, localMax, cornerMap);
			return cornerMap;
		}

		void getCorners(vector<Point>&points, double qualityLevel){
			Mat cornerMap = getCornerMap(qualityLevel);
			getCorners(points, cornerMap);
		}
		void getCorners(vector<Point>& points, const Mat& cornerMap){
			for (int y = 0; y < cornerMap.rows; y++){
				const uchar* rowPtr = cornerMap.ptr<uchar>(y);
				for (int x = 0; x < cornerMap.cols; x++){
					if (rowPtr[x]){
						points.push_back(Point(x, y));
					}
				}
			}
		}

		void dranOnImage(Mat& image, const vector<Point>&points, Scalar color = Scalar(255, 255, 255), int radius = 3, int thickness = 2){
			vector<Point>::const_iterator it = points.begin();
			while (it != points.end()){
				circle(image, *it, radius, color, thickness);
				++it;
			}
		}
};