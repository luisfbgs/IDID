#ifndef IDID_H
#define IDID_H

#include <opencv2/opencv.hpp>
#include "interpolations.hpp"

using namespace cv;

template<class T>
Mat_<T> columnMat(const Mat_<T> &m);

Mat_<uchar> IDID(const Mat_<uchar> &image, double scale, Itp interpolation = Itp::bilinear);

Mat_<uchar> splitIDID(const Mat_<uchar> &img, double scale, Itp interpolation = Itp::bilinear);


#endif