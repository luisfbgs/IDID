#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "interpolations.hpp"

using namespace cv;
using namespace std;

Mat_<double> bilinearMat(int rows, int cols, int nrows, int ncols){
	Mat_<double> H(nrows * ncols, rows * cols);
	double scaleX = 1. * nrows / rows, scaleY = 1. * ncols / cols;
	H = 0;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			int x = (.5 + i) * scaleX + .5, y = (.5 + j) * scaleY + .5;
			int nx = (.5 + i + 1) * scaleX + .5, ny = (.5 + j + 1) * scaleY + .5;
			int ni = min(i+1, rows-1), nj = min(j+1, cols-1);
			for(int x1 = i ? x : 0; x1 < min(nrows, nx); x1++){
				for(int y1 = j ? y : 0; y1 < min(ncols, ny); y1++){
					double x2 = x1 + .5, y2 = y1 + .5;
					int k = x1 * ncols + y1;
					H(k, i * cols + j) += double(ny - y2) / (ny - y) * (nx - x2) / (nx - x);
					H(k, ni * cols + j) += double(ny - y2) / (ny - y) * (x2 - x) / (nx - x);
					H(k, i * cols + nj) += double(y2 - y) / (ny - y) * (nx - x2) / (nx - x);
					H(k, ni * cols + nj) += double(y2 - y) / (ny - y) * (x2 - x) / (nx - x);
				}
			}
		}
	}
	return H.clone();
}

Mat_<uchar> bilinearScale(const Mat_<uchar> &img, double scale){
	int rows = img.rows;
	int cols = img.cols;
	int nrows = rows * scale, ncols = cols * scale;
	Mat_<uchar> result(nrows, ncols);
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			int x = (.5 + i) * scale + .5, y = (.5 + j) * scale + .5;
			int nx = (.5 + i + 1) * scale + .5, ny = (.5 + j + 1) * scale + .5;
			int ni = min(i+1, rows-1), nj = min(j+1, cols-1);
			for(int x1 = i ? x : 0; x1 < min(nrows, nx); x1++){
				for(int y1 = j ? y : 0; y1 < min(ncols, ny); y1++){
					double x2 = x1 + .5, y2 = y1 + .5;
					result(x1, y1) = round(
							double(ny - y2) / (ny - y) * (nx - x2) / (nx - x) * img(i, j) +
							double(ny - y2) / (ny - y) * (x2 - x) / (nx - x) * img(ni, j) +
							double(y2 - y) / (ny - y) * (nx - x2) / (nx - x) * img(i, nj) +
							double(y2 - y) / (ny - y) * (x2 - x) / (nx - x) * img(ni, nj));
				}
			}
		}
	}
	return result.clone();
}
