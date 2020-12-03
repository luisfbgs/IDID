#ifndef INTER_H
#define INTER_H

#include <opencv2/opencv.hpp>

using namespace cv;

enum Itp {
	bilinear,
	bicubic
};

Mat_<double> bilinearMat(int rows, int cols, int nrows, int ncols);

Mat_<uchar> bilinearScale(const Mat_<uchar> &img, double scale);

Mat_<uchar> directDownsample(const Mat_<uchar> &img, double scale);

template<class T>
Mat_<T> columnMat(const Mat_<T> &m);

#endif