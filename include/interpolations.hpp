#ifndef INTER_H
#define INTER_H

#include <opencv2/opencv.hpp>

using namespace cv;

Mat_<double> bilinearMat(int rows, int cols, int nrows, int ncols);

Mat_<unsigned char> bilinearScale(const Mat_<unsigned char> &img, double scale);

#endif