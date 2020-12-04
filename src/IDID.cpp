#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "IDID.hpp"
#include "interpolations.hpp"

using namespace cv;
using namespace std;

template<class T>
Mat_<T> columnMat(const Mat_<T> &m){
	int id = 0;
	Mat_<T> column(m.rows * m.cols, 1);
	for(int i = 0; i < m.rows; i++){
		for(int j = 0; j < m.cols; j++){
			column(id++, 0) = m(i, j);
		}
	}
	return column.clone();
}

Mat_<uchar> IDID(const Mat_<uchar> &image, double scale, Itp interpolation){
	int rows = max(1., image.rows / scale);
	int cols = max(1., image.cols / scale);
	Mat_<double> H;
	switch(interpolation){
		case(Itp::bicubic):
			H = bicubicMat(rows, cols, image.rows, image.cols);
			break;
		default:
			H = bilinearMat(rows, cols, image.rows, image.cols);
	}
	Mat_<double> Y = columnMat<uchar>(image);
	Mat_<double> HT;
	transpose(H, HT);
	Mat_<double> HHT = HT * H;
	Mat_<double> iv = HHT.inv();
	Mat_<double> res = iv * HT * Y;

	Mat_<uchar> newImg(rows, cols);
	int id = 0;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			newImg(i, j) = max(0, min(255, (int)round(res(id, 0))));
			id++;
		}
	}
	return newImg.clone();
}

Mat_<uchar> splitIDID(const Mat_<uchar> &img, double scale, Itp interpolation){
	int sz = 32;
	Mat_<uchar> result(img.rows, img.cols);
	result = 0;
	int lstx, lsty;
	lstx = lsty = 0;
	for(int i = 0; i < img.rows; i += sz){
		int stepx = 0;
		lsty = 0;
		for(int j = 0; j < img.cols; j += sz){
			int rows = min(sz, img.rows - i);
			int cols = min(sz, img.cols - j);
			Mat_<uchar> block = Mat_<uchar>(rows, cols);

			int bX = 0, bY = 0; 
			for(int x = i; x < i + rows; x++){
				for(int y = j; y < j + cols; y++){
					block(bX, bY) = img(x, y);
					bY++;
				}
				bX++;
				bY = 0;
			}
			block = IDID(block, scale, interpolation);
			
			int rx = lstx, ry = lsty;	
			for(int x = 0; x < block.rows; x++){
				for(int y = 0; y < block.cols; y++){
					result(rx, ry) = block(x, y);
					ry++;
				}
				rx++;
				ry = lsty;
			}
			lsty += block.cols;
			stepx = block.rows;
		}
		lstx += stepx;
	}
	Rect cropR(0, 0, lsty, lstx);
	return result(cropR).clone();
}