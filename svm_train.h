#pragma once
#include "stdafx.h"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv2/objdetect/objdetect.hpp>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
using namespace std;
using namespace cv;
void train();
void test();
void accuracy();
vector< float > get_svm_detector(const Ptr< ml::SVM >& svm);