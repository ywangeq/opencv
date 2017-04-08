
#include <stdio.h>  
#include <string.h>  
#include "cv.h"  
#include "cvaux.h"  
#include "highgui.h"  
#include "time.h"  
#include "iostream"

using namespace cv;
using namespace std;

////定义几个重要的全局变量  
IplImage ** faceImgArr = 0; // 指向训练人脸和测试人脸的指针（在学习和识别阶段指向不同）  
CvMat    *  personNumTruthMat = 0; // 人脸图像的ID号  
int nTrainFaces = 0; // 训练图像的数目  
int nEigens =0; // 自己取的主要特征值数目  
IplImage * pAvgTrainImg = 0; // 训练人脸数据的平均值  
IplImage ** eigenVectArr = 0; // 投影矩阵，也即主特征向量  
CvMat * eigenValMat = 0; // 特征值  
CvMat * projectedTrainFaceMat = 0; // 训练图像的投影  


//// 函数原型  
void learn();
void recognize();
void video_rec();
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int  loadFaceImgArray(char * filename);
void printUsage();



//主函数，主要包括学习和识别两个阶段，需要运行两次，通过命令行传入的参数区分  
int main()
{
	learn();  
	recognize();
	video_rec();

}


//学习阶段代码  
void learn()
{
	cout << "开始训练过程" << endl;

	//开始计时  
	int i, offset;

	//加载训练图像集  
	nTrainFaces = loadFaceImgArray("try/train.txt");
	if (nTrainFaces < 2)
	{
		fprintf(stderr,
			"Need 2 or more training faces\n"
			"Input file contains only %d\n", nTrainFaces);
		return;
	}

	// 进行主成分分析  
	doPCA();
	
	//将训练图集投影到子空间中  
	projectedTrainFaceMat = cvCreateMat(nTrainFaces, nEigens, CV_32FC1);
	offset = projectedTrainFaceMat->step / sizeof(float);
	for (i = 0; i<nTrainFaces; i++)
	{
		//int offset = i * nEigens;  
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			//projectedTrainFaceMat->data.fl + i*nEigens);  
			projectedTrainFaceMat->data.fl + i*offset);
	}

	//将训练阶段得到的特征值，投影矩阵等数据存为.xml文件，以备测试时使用  
	storeTrainingData();


}


//识别阶段代码  
void recognize()
{
	cout << "开始识别过程" << endl;

	// 测试人脸数  
	int i, nTestFaces = 0;

	// 训练阶段的人脸数 
	CvMat * trainPersonNumMat = 0;
	float * projectedTestFace = 0;

	// 加载测试图像，并返回测试人脸数  
	nTestFaces = loadFaceImgArray("try/test.txt");
	printf("%d test faces loaded\n", nTestFaces);

	// 加载保存在.xml文件中的训练结果  
	if (!loadTrainingData(&trainPersonNumMat))
		return;

	projectedTestFace = (float *)cvAlloc(nEigens*sizeof(float));
	for (i = 0; i<nTestFaces; i++)
	{
		int iNearest, nearest, truth;

		//将测试图像投影到子空间中  
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			projectedTestFace);

		iNearest = findNearestNeighbor(projectedTestFace);
		truth = personNumTruthMat->data.i[i];
		nearest = trainPersonNumMat->data.i[iNearest];

		printf("nearest = %d, Truth = %d\n", nearest, truth);
	}

	
}


//加载保存过的训练结果  
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;


	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_READ);
	if (!fileStorage)
	{
		fprintf(stderr, "Can't open facedata.xml\n");
		return 0;
	}

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for (i = 0; i<nEigens; i++)
	{
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}


	cvReleaseFileStorage(&fileStorage);

	return 1;
}



//存储训练结果  
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;


	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_WRITE);

	//存储特征值，投影矩阵，平均矩阵等训练结果  
	cvWriteInt(fileStorage, "nEigens", nEigens);
	cvWriteInt(fileStorage, "nTrainFaces", nTrainFaces);
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0, 0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0, 0));
	for (i = 0; i<nEigens; i++)
	{
		char varname[200];
		sprintf(varname, "eigenVect_%d", i);
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0, 0));
		cvNormalize(eigenVectArr[i], eigenVectArr[i], 255, 0, CV_L2, 0);
		cvNamedWindow("demo", CV_WINDOW_AUTOSIZE);
		cvShowImage("demo", eigenVectArr[i]);
		cvWaitKey(100);

	}
	cvNormalize(pAvgTrainImg, pAvgTrainImg, 255, 0, CV_L1, 0);
	cvNamedWindow("demo", CV_WINDOW_AUTOSIZE);
	cvShowImage("demo", pAvgTrainImg);
	cvWaitKey(100);



	cvReleaseFileStorage(&fileStorage);
}



//寻找最接近的图像  
int findNearestNeighbor(float * projectedTestFace)
{

	//定义最小距离，并初始化为无穷大
	double leastDistSq = DBL_MAX, accuracy;
	int i, iTrain, iNearest = 0;
	double a[500];

	for (iTrain = 0; iTrain<nTrainFaces; iTrain++)
	{
		double distSq = 0;

		for (i = 0; i<nEigens; i++)
		{
			float d_i =
				projectedTestFace[i] -
				projectedTrainFaceMat->data.fl[iTrain*nEigens + i];

			// Mahalanobis算法计算的距离
			//distSq += d_i*d_i; // Euclidean算法计算的距离  
			distSq += d_i*d_i / eigenValMat->data.fl[i];

		}
		a[iTrain] = distSq;

		if (distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}
	//求阈值
	double max = a[0], threshold;
	int j;
	for (j = 1; j<10; j++)
	{
		if (max<a[j])
			max = a[j];
		else
			max = max;
	}
	threshold = max / 2;
	//求相似率
	accuracy = 1 - leastDistSq / threshold;
	cout << "相似率为:" << accuracy << endl;
	return iNearest;
}



//主成分分析  
void doPCA()
{
	int i;

	//终止算法准则
	CvTermCriteria calcLimit;

	//构造图像
	CvSize faceImgSize;

	// 自己设置主特征值个数  
	nEigens = nTrainFaces - 1;

	//分配特征向量存储空间  
	faceImgSize.width = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;

	//分配个数为主特征值个数
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*)* nEigens);
	for (i = 0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	//分配主特征值存储空间  
	eigenValMat = cvCreateMat(1, nEigens, CV_32FC1);

	// 分配平均图像存储空间  
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// 设定PCA分析结束条件  
	calcLimit = cvTermCriteria(CV_TERMCRIT_ITER, nEigens, 1);

	// 计算平均图像，特征值，特征向量  
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);

	//归一化大小
	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}



//加载txt文件的列举的图像  
int loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces = 0;


	if (!(imgListFile = fopen(filename, "r")))
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// 统计人脸数  
	while (fgets(imgFilename, 512, imgListFile)) ++nFaces;
	rewind(imgListFile);

	// 分配人脸图像存储空间和人脸ID号存储空间  
	faceImgArr = (IplImage **)cvAlloc(nFaces*sizeof(IplImage *));
	personNumTruthMat = cvCreateMat(1, nFaces, CV_32SC1);

	for (iFace = 0; iFace<nFaces; iFace++)
	{
		// 从文件中读取序号和人脸名称  
		fscanf(imgListFile,
			"%d %s", personNumTruthMat->data.i + iFace, imgFilename);

		// 加载人脸图像  
		faceImgArr[iFace] = cvLoadImage(imgFilename, 0);

		if (!faceImgArr[iFace])
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
		cvNamedWindow("demo", CV_WINDOW_AUTOSIZE);
		cvShowImage("demo", faceImgArr[iFace]);
		cvWaitKey(100);
	}

	fclose(imgListFile);

	return nFaces;
}



//  
void printUsage()
{
	printf("Usage: eigenface <command>\n",
		"  Valid commands are\n"
		"    train\n"
		"    test\n");
}


void video_rec() //没写完
{
	// 测试人脸数  
	int i, nTestFaces = 0;

	// 训练阶段的人脸数 
	CvMat * trainPersonNumMat = 0;
	float * projectedTestFace = 0;

	// 加载测试图像，并返回测试人脸数  
	VideoCapture cap(0);    //打开默认摄像头
	Mat frame;
	Mat edges;
	Mat gray;

	CascadeClassifier cascade;
	bool stop = false;
	cascade.load("haarcascade_frontalface_alt.xml");
	while (!stop)
	{ 
		cap >> frame;
		vector<Rect> faces(0);
		cvtColor(frame, gray, CV_BGR2GRAY);
		resize(gray, gray,Size(150,200), INTER_LINEAR);
		equalizeHist(gray, gray);

	}//nTestFaces = loadFaceImgArray("try/test.txt");
	printf("%d test faces loaded\n", nTestFaces);

	// 加载保存在.xml文件中的训练结果  
	if (!loadTrainingData(&trainPersonNumMat))
		return;

	projectedTestFace = (float *)cvAlloc(nEigens*sizeof(float));
	for (i = 0; i<nTestFaces; i++)
	{
		int iNearest, nearest, truth;

		//将测试图像投影到子空间中  
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			projectedTestFace);

		iNearest = findNearestNeighbor(projectedTestFace);
		truth = personNumTruthMat->data.i[i];
		nearest = trainPersonNumMat->data.i[iNearest];

		printf("nearest = %d, Truth = %d\n", nearest, truth);
	}


}
