
#include <stdio.h>  
#include <string.h>  
#include "cv.h"  
#include "cvaux.h"  
#include "highgui.h"  
#include "time.h"  
#include "iostream"

using namespace cv;
using namespace std;

////���弸����Ҫ��ȫ�ֱ���  
IplImage ** faceImgArr = 0; // ָ��ѵ�������Ͳ���������ָ�루��ѧϰ��ʶ��׶�ָ��ͬ��  
CvMat    *  personNumTruthMat = 0; // ����ͼ���ID��  
int nTrainFaces = 0; // ѵ��ͼ�����Ŀ  
int nEigens =0; // �Լ�ȡ����Ҫ����ֵ��Ŀ  
IplImage * pAvgTrainImg = 0; // ѵ���������ݵ�ƽ��ֵ  
IplImage ** eigenVectArr = 0; // ͶӰ����Ҳ������������  
CvMat * eigenValMat = 0; // ����ֵ  
CvMat * projectedTrainFaceMat = 0; // ѵ��ͼ���ͶӰ  


//// ����ԭ��  
void learn();
void recognize();
void video_rec();
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int  loadFaceImgArray(char * filename);
void printUsage();



//����������Ҫ����ѧϰ��ʶ�������׶Σ���Ҫ�������Σ�ͨ�������д���Ĳ�������  
int main()
{
	learn();  
	recognize();
	video_rec();

}


//ѧϰ�׶δ���  
void learn()
{
	cout << "��ʼѵ������" << endl;

	//��ʼ��ʱ  
	int i, offset;

	//����ѵ��ͼ��  
	nTrainFaces = loadFaceImgArray("try/train.txt");
	if (nTrainFaces < 2)
	{
		fprintf(stderr,
			"Need 2 or more training faces\n"
			"Input file contains only %d\n", nTrainFaces);
		return;
	}

	// �������ɷַ���  
	doPCA();
	
	//��ѵ��ͼ��ͶӰ���ӿռ���  
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

	//��ѵ���׶εõ�������ֵ��ͶӰ��������ݴ�Ϊ.xml�ļ����Ա�����ʱʹ��  
	storeTrainingData();


}


//ʶ��׶δ���  
void recognize()
{
	cout << "��ʼʶ�����" << endl;

	// ����������  
	int i, nTestFaces = 0;

	// ѵ���׶ε������� 
	CvMat * trainPersonNumMat = 0;
	float * projectedTestFace = 0;

	// ���ز���ͼ�񣬲����ز���������  
	nTestFaces = loadFaceImgArray("try/test.txt");
	printf("%d test faces loaded\n", nTestFaces);

	// ���ر�����.xml�ļ��е�ѵ�����  
	if (!loadTrainingData(&trainPersonNumMat))
		return;

	projectedTestFace = (float *)cvAlloc(nEigens*sizeof(float));
	for (i = 0; i<nTestFaces; i++)
	{
		int iNearest, nearest, truth;

		//������ͼ��ͶӰ���ӿռ���  
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


//���ر������ѵ�����  
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



//�洢ѵ�����  
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;


	fileStorage = cvOpenFileStorage("facedata.xml", 0, CV_STORAGE_WRITE);

	//�洢����ֵ��ͶӰ����ƽ�������ѵ�����  
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



//Ѱ����ӽ���ͼ��  
int findNearestNeighbor(float * projectedTestFace)
{

	//������С���룬����ʼ��Ϊ�����
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

			// Mahalanobis�㷨����ľ���
			//distSq += d_i*d_i; // Euclidean�㷨����ľ���  
			distSq += d_i*d_i / eigenValMat->data.fl[i];

		}
		a[iTrain] = distSq;

		if (distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}
	//����ֵ
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
	//��������
	accuracy = 1 - leastDistSq / threshold;
	cout << "������Ϊ:" << accuracy << endl;
	return iNearest;
}



//���ɷַ���  
void doPCA()
{
	int i;

	//��ֹ�㷨׼��
	CvTermCriteria calcLimit;

	//����ͼ��
	CvSize faceImgSize;

	// �Լ�����������ֵ����  
	nEigens = nTrainFaces - 1;

	//�������������洢�ռ�  
	faceImgSize.width = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;

	//�������Ϊ������ֵ����
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*)* nEigens);
	for (i = 0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	//����������ֵ�洢�ռ�  
	eigenValMat = cvCreateMat(1, nEigens, CV_32FC1);

	// ����ƽ��ͼ��洢�ռ�  
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// �趨PCA������������  
	calcLimit = cvTermCriteria(CV_TERMCRIT_ITER, nEigens, 1);

	// ����ƽ��ͼ������ֵ����������  
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

	//��һ����С
	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}



//����txt�ļ����оٵ�ͼ��  
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

	// ͳ��������  
	while (fgets(imgFilename, 512, imgListFile)) ++nFaces;
	rewind(imgListFile);

	// ��������ͼ��洢�ռ������ID�Ŵ洢�ռ�  
	faceImgArr = (IplImage **)cvAlloc(nFaces*sizeof(IplImage *));
	personNumTruthMat = cvCreateMat(1, nFaces, CV_32SC1);

	for (iFace = 0; iFace<nFaces; iFace++)
	{
		// ���ļ��ж�ȡ��ź���������  
		fscanf(imgListFile,
			"%d %s", personNumTruthMat->data.i + iFace, imgFilename);

		// ��������ͼ��  
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


void video_rec()
{
	// ����������  
	int i, nTestFaces = 0;

	// ѵ���׶ε������� 
	CvMat * trainPersonNumMat = 0;
	float * projectedTestFace = 0;

	// ���ز���ͼ�񣬲����ز���������  
	VideoCapture cap(0);    //��Ĭ������ͷ
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

	// ���ر�����.xml�ļ��е�ѵ�����  
	if (!loadTrainingData(&trainPersonNumMat))
		return;

	projectedTestFace = (float *)cvAlloc(nEigens*sizeof(float));
	for (i = 0; i<nTestFaces; i++)
	{
		int iNearest, nearest, truth;

		//������ͼ��ͶӰ���ӿռ���  
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