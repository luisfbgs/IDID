#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "interpolations.hpp"

using namespace cv;
using namespace std;

Mat_<uchar> directDownsample(const Mat_<uchar> &img, double scale){
	Mat_<uchar> result;
	resize(img, result, Size(), 1 / scale, 1 / scale, INTER_NEAREST);
	return result.clone();
}

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

Mat_<uchar> IDID(const Mat_<uchar> &image, double scale, InterpolationFlags interpolation = INTER_LINEAR_EXACT, const Mat_<uchar> &outBlock = Mat_<uchar>(0, 0)){
	int rows = max(1., image.rows / scale);
	int cols = max(1., image.cols / scale);
	Mat_<double> H;
	switch(INTER_LINEAR_EXACT){
		default:
			H = bilinearMat(rows, cols, image.rows, image.cols);
	}
	Mat_<double> HT;
	Mat_<double> Y = columnMat<uchar>(image);
	transpose(H, HT);
	Mat_<double> HHT = HT * H;
	Mat_<double> iv = HHT.inv();
	iv = iv * HT;
	Mat_<double> res = iv * Y;

	Mat_<uchar> newImg(rows, cols);
	int id = 0;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			newImg(i, j) = min(255, (int)round(res(id++, 0)));
		}
	}
	return newImg.clone();
}

Mat_<uchar> splitIDID(const Mat_<uchar> &img, double scale){
	int sz = 16;
	Mat_<uchar> result(img.rows, img.cols);
	int lstx = 0, lsty = 0;
	for(int i = 0; i < img.rows; i += sz){
		int stepx = 0;
		lsty = 0;
		for(int j = 0; j < img.cols; j += sz){
			int rows = min(sz, img.rows - i);
			int cols = min(sz, img.cols - j);
			Mat_<uchar> block = Mat_<uchar>(rows, cols);
			int oRows = rows + (i != 0) + (rows + i < img.rows);
			int oCols = cols + (j != 0) + (cols + j < img.cols);
			Mat_<uchar> outBlock(oRows, oCols);
			int bX = 0, bY = 0; 
			for(int x = i; x < i + rows; x++){
				for(int y = j; y < j + cols; y++){
					block(bX, bY) = img(x, y);
					bY++;
				}
				bX++;
				bY = 0;
			}

			bX = bY = 0; 
			for(int x = max(0, i-1); x < i + oRows; x++){
				for(int y = j; y < j + oCols; y++){
					outBlock(bX, bY) = img(x, y);
					bY++;
				}
				bX++;
				bY = 0;
			}

			block = IDID(block, scale, INTER_LINEAR_EXACT, outBlock);
			
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
	Rect cropR(0, 0, lstx, lsty);
	return result(cropR).clone();
}

int main(){
	Mat_<uchar> image;
	image = imread("miniLenna.png", 0);
	//imwrite("B.png", bilinearScale(IDID(image, 5), 5));
	Mat_<uchar> res = bilinearScale(splitIDID(image, 2), 2);
	imwrite("C.png", res);
	res = splitIDID(image, 2);
	imwrite("D.png", res);
	return 0;
}
