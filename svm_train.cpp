#include "stdafx.h"
#include "svm_train.h"
#include "fhog.hpp"
void train() {
	int descriptorDim;

	string buffer;
	string trainImg;
	vector<string> posSamples;
	vector<string> negSamples;
	int posSampleNum;
	int negSampleNum;
	string PosPath = "C:\\Users\\luo\\Desktop\\carVideo\\pos\\";
	string NegPath = "C:\\Users\\luo\\Desktop\\carVideo\\neg\\";
	string namepos = PosPath + "names.txt";
	string nameneg = NegPath + "names.txt";

	ifstream fInPos(namepos);
	ifstream fInNeg(nameneg);

	while (fInPos)
	{
		if (getline(fInPos, buffer))
			posSamples.push_back(PosPath + buffer);
	}
	posSampleNum = posSamples.size();
	fInPos.close();

	while (fInNeg)
	{
		if (getline(fInNeg, buffer))
			negSamples.push_back(NegPath + buffer);
	}
	negSampleNum = negSamples.size();
	fInNeg.close();

	Mat sampleFeatureMat;
	Mat sampleLabelMat;

	sampleFeatureMat = Mat::zeros(posSampleNum + negSampleNum, 1116, CV_32FC1);
	sampleLabelMat = Mat::zeros(posSampleNum + negSampleNum, 1, CV_32S);

	//HOGDescriptor * hog = new HOGDescriptor(cvSize(32, 32), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);

	for (int i = 0; i < posSampleNum; i++)
	{
		vector<float> descriptor;
		Mat inputImg = imread(posSamples[i]);
		cout << "processing " << i << "/" << posSampleNum << " " << posSamples[i] << endl;
		Size dsize = Size(32, 32);
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputImg, trainImg, dsize);
		cvtColor(trainImg,trainImg, CV_BGR2GRAY);
/////////////////////fhog//////////////////
		IplImage Img = trainImg;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&Img, 4, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);

		int sizeX, sizeY, nfeatures;
		sizeX = map->sizeX;
		sizeY = map->sizeY;
		nfeatures = map->numFeatures;

		for (int k = 0; k< sizeY; ++k)
		{
			for (int t = 0; t < sizeX; ++t)
			{
				for (int n = 0; n < nfeatures; ++n)
				{
					descriptor.push_back(map->map[(k*sizeX + t)*nfeatures + n]);
				}
			}
		}
///////////////////////////////////////////////////////////////

		//hog->compute(trainImg, descriptor);
		int descriptorDim = descriptor.size();


		for (int j = 0; j < descriptorDim; j++)
		{
			sampleFeatureMat.at<float>(i, j) = descriptor[j];
		}

		sampleLabelMat.at<int>(i,0) = 1;
	}

	cout << "extract posSampleFeature done" << endl;

	for (int i = 0; i < negSampleNum; i++)
	{
		vector<float> descriptor;
		Mat inputImg = imread(negSamples[i]);
		cout << "processing " << i << "/" << negSampleNum << " " << negSamples[i] << endl;
		Size dsize = Size(32, 32);
		Mat trainImg = Mat(dsize, CV_32S);
		resize(inputImg, trainImg, dsize);
		cvtColor(trainImg, trainImg, CV_BGR2GRAY);
		//hog->compute(trainImg, descriptor);
//////////fhog//////////////////////////
		IplImage Img = trainImg;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&Img, 4, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);

		int sizeX, sizeY, nfeatures;
		sizeX = map->sizeX;
		sizeY = map->sizeY;
		nfeatures = map->numFeatures;

		for (int k = 0; k< sizeY; ++k)
		{
			for (int t = 0; t < sizeX; ++t)
			{
				for (int n = 0; n < nfeatures; ++n)
				{
					descriptor.push_back(map->map[(k*sizeX + t)*nfeatures + n]);
				}
			}
		}
///////////////////////////////////////
		descriptorDim = descriptor.size();

		for (int j = 0; j < descriptorDim; j++)
		{
			sampleFeatureMat.at<float>(posSampleNum + i, j) = descriptor[j];
		}

		sampleLabelMat.at<int>(posSampleNum + i,0) = -1;
	}

	cout << "extract negSampleFeature done" << endl;

	

	ofstream foutFeature("SampleFeatureMat.txt");
	for (int i = 0; i < posSampleNum + negSampleNum; i++)
	{
		for (int j = 0; j < descriptorDim; j++)
		{
			foutFeature << sampleFeatureMat.at<float>(i, j) << " ";
		}
		foutFeature << "\n";
	}
	foutFeature.close();
	cout << "output posSample and negSample Feature done" << endl;

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setType(ml::SVM::C_SVC);
	svm->setKernel(ml::SVM::LINEAR);
	//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON));

	// CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
	//  CvSVMParams params(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);  //这里一定要注意，LINEAR代表的是线性核，RBF代表的是高斯核，如果要用opencv自带的detector必须用线性核，如果自己写，或者只是判断是否为车脸的2分类问题则可以用RBF，在此应用环境中线性核的性能还是不错的
	cout << "SVM Training Start..." << endl;
	svm->train(sampleFeatureMat, ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);
	//  SVM.train_auto(sampleFeatureMat, sampleFeatureMat, Mat(), Mat(), params);
	//  SVM.save("SVM_Model.xml");
	svm->save("SVM_Model.xml");
	cout << "SVM Training Complete" << endl;

}

void accuracy() {
	int descriptorDim;

	string buffer;
	vector<string> posSamples;
	vector<string> negSamples;
	int posSampleNum;
	int negSampleNum;
	int testSampleNum;
	string PosPath = "C:\\Users\\luo\\Desktop\\carVideo\\testpos\\";
	string NegPath = "C:\\Users\\luo\\Desktop\\carVideo\\testneg\\";
	string namepos = PosPath + "names.txt";
	string nameneg = NegPath + "names.txt";

	ifstream fInPos(namepos);
	ifstream fInNeg(nameneg);

	while (fInPos)
	{
		if (getline(fInPos, buffer))
			posSamples.push_back(PosPath + buffer);
	}
	posSampleNum = posSamples.size();
	fInPos.close();

	while (fInNeg)
	{
		if (getline(fInNeg, buffer))
			negSamples.push_back(NegPath + buffer);
	}
	negSampleNum = negSamples.size();
	fInNeg.close();

	Mat sampleFeatureMat;
	Mat sampleLabelMat;
	Mat result;

	sampleFeatureMat = Mat::zeros(posSampleNum + negSampleNum, 1116, CV_32FC1);
	sampleLabelMat = Mat::zeros(posSampleNum + negSampleNum, 1, CV_32S);
	result = Mat::zeros(posSampleNum + negSampleNum, 1, CV_32FC1);

	HOGDescriptor * hog = new HOGDescriptor(cvSize(32, 32), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
	

	for (int i = 0; i < posSampleNum; i++)
	{
		vector<float> descriptor;
		Mat inputImg = imread(posSamples[i]);
		Size dsize = Size(32, 32);
		Mat testImg = Mat(dsize, CV_32S);
		resize(inputImg, testImg, dsize);
		cvtColor(testImg, testImg, CV_BGR2GRAY);
		/////////////////////fhog//////////////////
		IplImage Img = testImg;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&Img, 4, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);

		int sizeX, sizeY, nfeatures;
		sizeX = map->sizeX;
		sizeY = map->sizeY;
		nfeatures = map->numFeatures;

		for (int k = 0; k< sizeY; ++k)
		{
			for (int t = 0; t < sizeX; ++t)
			{
				for (int n = 0; n < nfeatures; ++n)
				{
					descriptor.push_back(map->map[(k*sizeX + t)*nfeatures + n]);
				}
			}
		}
		///////////////////////////////////////////////////////////////

		//hog->compute(testImg, descriptor);
		descriptorDim = descriptor.size();

		
		for (int j = 0; j < descriptorDim; j++)
		{
			sampleFeatureMat.at<float>(i, j) = descriptor[j];
		}

		sampleLabelMat.at<int>(i, 0) = 1;
	}

	cout << "extract posSampleFeature done" << endl;

	for (int i = 0; i < negSampleNum; i++)
	{
		vector<float> descriptor;
		Mat inputImg = imread(negSamples[i]);
		Size dsize = Size(32, 32);
		Mat testImg = Mat(dsize, CV_32S);
		resize(inputImg, testImg, dsize);
		cvtColor(testImg, testImg, CV_BGR2GRAY);
		/////////////////////fhog//////////////////
		IplImage Img = testImg;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&Img, 4, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);

		int sizeX, sizeY, nfeatures;
		sizeX = map->sizeX;
		sizeY = map->sizeY;
		nfeatures = map->numFeatures;

		for (int k = 0; k< sizeY; ++k)
		{
			for (int t = 0; t < sizeX; ++t)
			{
				for (int n = 0; n < nfeatures; ++n)
				{
					descriptor.push_back(map->map[(k*sizeX + t)*nfeatures + n]);
				}
			}
		}
		///////////////////////////////////////////////////////////////
		//hog->compute(testImg, descriptor);
		descriptorDim = descriptor.size();

		for (int j = 0; j < descriptorDim; j++)
		{
			sampleFeatureMat.at<float>(posSampleNum + i, j) = descriptor[j];
		}

		sampleLabelMat.at<int>(posSampleNum + i, 0) = -1;
	}

	Ptr<ml::SVM> svm = ml::SVM::load("SVM_Model.xml");
     svm->predict(sampleFeatureMat,result);
	float posRight = 0,negRight = 0, Right=0;
	int posSum =0,negSum=0;
	for (int i = 0; i <  posSampleNum + negSampleNum; i++) {
		cout <<"real:" <<sampleLabelMat.at<int>(i, 0)<<"test:"<<result.at<float>(i, 0) << endl;
		if (sampleLabelMat.at<int>(i, 0) == 1) {
			posSum++;
			if (sampleLabelMat.at<int>(i, 0) == result.at<float>(i, 0))
				posRight++;
		}
		else {
			negSum++;
			if (sampleLabelMat.at<int>(i, 0) == result.at<float>(i, 0))
				negRight++;
		}
	}
	Right = (posRight + negRight) / (posSum + negSum);
	posRight =  posRight/ posSum;
	negRight = negRight / negSum;
	
	cout << "class finish" << endl;
	cout << "positive accuracy:" << posRight<<endl;
	cout << "negtive accuracy:" << negRight<< endl;
	cout << "accuracy" << Right << endl;
}


vector< float > get_svm_detector(const Ptr< ml::SVM >& svm)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	vector< float > hog_detector(sv.cols + 1);
	//memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	//hog_detector[sv.cols] = (float)-rho;
	for (int i = 0; i < sv.cols; i++) {
		hog_detector[i] = -sv.at<float>(0, i);
	}
	hog_detector[sv.cols] = (float)rho;
	return hog_detector;
}

void test() {

	vector<float> myDetector;
	Ptr<ml::SVM> svm = ml::SVM::load("SVM_Model.xml");

	HOGDescriptor myHOG(Size(32, 32), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	myHOG.setSVMDetector(get_svm_detector(svm));//设置检测子


	/*vector<string> testSamples;
	string buffer;

	test part
	ifstream fInTest("D:\\DataSet\\CarFaceDataSet\\testSample.txt");
	while (fInTest)
	{
		if (getline(fInTest, buffer))
		{
			testSamples.push_back(buffer);
		}
	}
	int testSampleNum = testSamples.size();
	fInTest.close();
*/
	/*for (int i = 0; i < testSamples.size(); i++)
	{
		Mat testImg = imread(testSamples[i]);
		Size dsize = Size(320, 240);
		Mat testImgNorm(dsize, CV_32S);
		resize(testImg, testImgNorm, dsize);*/

		Mat testImgNorm = imread("C:\\Users\\luo\\Desktop\\carVideo\\268.png");
		//resize(testImgNorm, testImgNorm, Size(540, 512));

		vector<Rect> found, foundFiltered;
		cout << "MultiScale detect " << endl;
		myHOG.detectMultiScale(testImgNorm, found, 0, Size(8, 8), Size(0, 0), 1.2, 1, true);
	/*	vector< double > foundWeights;
		myHOG.detectMultiScale(testImgNorm, found, foundWeights);*/
		cout << "Detected Rect Num:" << found.size() << endl;

		for (int i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j = 0;
			for (; j < found.size(); j++)
			{
				if (i != j && (r & found[j]) == r)
				{
					break;
				}
			}
			if (j == found.size())
				foundFiltered.push_back(r);
		}
		for (int i = 0; i < foundFiltered.size(); i++)
		{
			Rect r = foundFiltered[i];
			rectangle(testImgNorm, r.tl(), r.br(), Scalar(0, 255, 0), 1);
		}

		//for (int i = 0; i < found.size(); i++)
		//{
		//	Rect r = found[i];
		//	rectangle(testImgNorm, r.tl(), r.br(), Scalar(0, 255, 0), 1);
		//}

		imshow("test", testImgNorm);
		waitKey(0);
	/*}

	system("pause");*/


}
