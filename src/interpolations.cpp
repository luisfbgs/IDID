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
			double x = (.5 + i) * scaleX, y = (.5 + j) * scaleY;
			double nx = (.5 + i + 1) * scaleX, ny = (.5 + j + 1) * scaleY;
			int ni = min(i+1, rows-1), nj = min(j+1, cols-1);
			if(!i) x = 0;
			if(!j) y = 0;
			for(int x1 = x; x1 < min(nrows, (int)ceil(nx)); x1++){
				for(int y1 = y; y1 < min(ncols, (int)ceil(ny)); y1++){
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
			double x = (.5 + i) * scale, y = (.5 + j) * scale;
			double nx = (.5 + i + 1) * scale, ny = (.5 + j + 1) * scale;
			int ni = min(i+1, rows-1), nj = min(j+1, cols-1);
			if(!i) x = 0;
			if(!j) y = 0;
			for(int x1 = x; x1 < min(nrows, (int)ceil(nx)); x1++){
				for(int y1 = y; y1 < min(ncols, (int)ceil(ny)); y1++){
					double x2 = x1 + .5, y2 = y1 + .5;
					result(x1, y1) = min(255., max(0., round(
							double(ny - y2) / (ny - y) * (nx - x2) / (nx - x) * img(i, j) +
							double(ny - y2) / (ny - y) * (x2 - x) / (nx - x) * img(ni, j) +
							double(y2 - y) / (ny - y) * (nx - x2) / (nx - x) * img(i, nj) +
							double(y2 - y) / (ny - y) * (x2 - x) / (nx - x) * img(ni, nj))));
				}
			}
		}
	}
	return result.clone();
}

void bicubicCoe(double mat[4][4], double x, double y){
	double x2 = x * x;
	double x3 = x2 * x;
	double y2 = y * y;
	double y3 = y2 * y;
	mat[0][0] = (.25* y * x -.5* y2 * x + .25* y3 * x -.5* y * x2 + y2 * x2 -.5* y3 * x2 + .25* y * x3 -.5* y2 * x3 + .25* y3 * x3);
	mat[0][1] = (-.5 * x + 1.25 * y2 * x - .75 * y3 * x + x2 - 2.5 * y2 * x2 + 1.5 * y3 * x2 -.5 * x3 + 1.25 * y2 * x3 - .75 * y3 * x3);
	mat[0][2] = (-.25 * y * x - y2 * x + .75 * y3 * x + .5 * y * x2 + 2 * y2 * x2 - 1.5 * y3 * x2 - .25 * y * x3 - y2 * x3 + .75 * y3 * x3);
	mat[0][3] = (+.25 * y2 * x - .25 * y3 * x - .5 * y2 * x2 + .5 * y3 * x2 + .25 * y2 * x3 - .25 * y3 * x3);
	mat[1][0] = (-.5 * y + y2-.5 * y3 + 1.25 * y * x2 - 2.5 * y2 * x2 + 1.25 * y3 * x2 - .75 * y * x3 + 1.5 * y2 * x3 - .75 * y3 * x3);
	mat[1][1] = (1 - 2.5 * y2 + 1.5 * y3 - 2.5 * x2 + 6.25 * y2 * x2 - 3.75 * y3 * x2 + 1.5 * x3 - 3.75 * y2 * x3 + 2.25 * y3 * x3);
	mat[1][2] = (.5 * y + 2 * y2- 1.5 * y3 - 1.25 * y * x2 - 5 * y2 * x2 + 3.75 * y3 * x2 + .75 * y * x3 + 3 * y2 * x3 - 2.25 * y3 * x3);
	mat[1][3] = (-.5 * y2 + .5 * y3 + 1.25 * y2 * x2 - 1.25 * y3 * x2 - .75 * y2 * x3 + .75 * y3 * x3);
	mat[2][0] = (-.25 * y * x + .5 * y2 * x - .25 * y3 * x - y * x2 + 2 * y2 * x2 - y3 * x2 + .75 * y * x3 - 1.5 * y2 * x3 + .75 * y3 * x3);
	mat[2][1] = (+.5 * x - 1.25 * y2 * x + .75 * y3 * x + 2 * x2 - 5 * y2 * x2 + 3 * y3 * x2 - 1.5 * x3 + 3.75 * y2 * x3 - 2.25 * y3 * x3);
	mat[2][2] = (+.25 * y * x + y2 * x - .75 * y3 * x + y * x2 + 4 * y2 * x2 - 3 * y3 * x2 - .75 * y * x3 - 3 * y2 * x3 + 2.25 * y3 * x3);
	mat[2][3] = (-.25 * y2 * x + .25 * y3 * x - y2 * x2 + y3 * x2 + .75 * y2 * x3 - .75 * y3 * x3);
	mat[3][0] = (+.25 * y * x2 - .5 * y2 * x2 + .25 * y3 * x2 - .25 * y * x3 + .5 * y2 * x3 - .25 * y3 * x3);
	mat[3][1] = (-.5 * x2 + 1.25 * y2 * x2 - .75 * y3 * x2 + .5 * x3- 1.25 * y2 * x3 + .75 * y3 * x3);
	mat[3][2] = (-.25 * y * x2 - y2 * x2 + .75 * y3 * x2 + .25 * y * x3+ y2 * x3 - .75 * y3 * x3);
	mat[3][3] = (+.25 * y2 * x2 - .25 * y3 * x2 - .25 * y2 * x3 + .25 * y3 * x3);
}

Mat_<uchar> bicubicScale(const Mat_<uchar> &img, double scale){
	int rows = img.rows;
	int cols = img.cols;
	int nrows = rows * scale, ncols = cols * scale;
	Mat_<uchar> result(nrows, ncols);
	result = 0;
	double coe[4][4];
	double mat[4][4];
	memset(mat, 0, sizeof mat);

	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			double x = (.5 + i) * scale, y = (.5 + j) * scale;
			double nx = (.5 + i + 1) * scale, ny = (.5 + j + 1) * scale;
			if(!i) x = 0;
			if(!j) y = 0;
			
			int mx = 0, my = 0;
			for(int a = i-1; a <= i+2; a++){
				for(int b = j-1; b <= j+2; b++){
					mat[mx][my] = (a >= 0 && b >= 0 && a < rows && b < cols) ? img(a, b) : img(i, j);
					my++;
				}
				mx++;
				my = 0;
			}

			for(int x1 = x; x1 < min(nrows, (int)ceil(nx)); x1++){
				for(int y1 = y; y1 < min(ncols, (int)ceil(ny)); y1++){
					double x2 = x1 + .5, y2 = y1 + .5;
					bicubicCoe(coe, (x2-x) / (nx - x), (y2-y) / (ny - y));
					double aux = 0;
					for(int a = 0; a < 4; a++){
						for(int b = 0; b < 4; b++){
							aux += coe[a][b] * mat[a][b];
						}
					}
					result(x1, y1) = min(255., max(0., aux));
				}
			}
		}
	}

	for(int i = 0; i < nrows; i++){
		for(int j = 0; j < ncols; j++){
			result(i, j) = min(max(0, (int)result(i, j)), 255);
		}
	}

	return result.clone();
}


Mat_<uchar> directDownsample(const Mat_<uchar> &img, double scale){
	Mat_<uchar> result;
	resize(img, result, Size(), 1 / scale, 1 / scale, INTER_AREA);
	return result.clone();
}