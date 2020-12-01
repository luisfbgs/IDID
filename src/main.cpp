#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "interpolations.hpp"

using namespace cv;
using namespace std;

Mat_<unsigned char> directDownsample(const Mat_<unsigned char> &img, double scale){
	Mat_<unsigned char> result;
	resize(img, result, Size(), 1 / scale, 1 / scale, INTER_NEAREST);
	return result;
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
	return column;
}

Mat_<unsigned char> IDID(const Mat_<unsigned char> &image, double scale, InterpolationFlags interpolation = INTER_LINEAR_EXACT){
	int rows = max(1., image.rows / scale);
	int cols = max(1., image.cols / scale);
	Mat_<double> H;
	switch(INTER_LINEAR_EXACT){
		default:
			H = bilinearMat(rows, cols, image.rows, image.cols);
	}
	Mat_<double> HT;
	Mat_<double> Y = columnMat<unsigned char>(image);
	transpose(H, HT);
	Mat_<double> HHT = HT * H;
	Mat_<double> iv = HHT.inv();
	iv = iv * HT;
	Mat_<double> res = iv * Y;

	Mat_<unsigned char> newImg(rows, cols);
	int id = 0;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			newImg(i, j) = min(255, (int)round(res(id++, 0)));
		}
	}
	return newImg;
}

Mat_<unsigned char> splitIDID(const Mat_<unsigned char> &img, double scale){
	int sz = min(img.rows, min(img.cols, int(scale * 5)));
	Mat_<unsigned char> result(img);
	int lstx = 0, lsty = 0;
	for(int i = 0; i < img.rows; i += sz){
		int stepx = 0;
		lsty = 0;
		for(int j = 0; j < img.cols; j += sz){
			int rows = min(sz, img.rows - i);
			int cols = min(sz, img.cols - j);
			Mat_<unsigned char> block(rows, cols);

			int bX = 0, bY = 0; 
			for(int x = i; x < min(img.rows, i + sz); x++){
				for(int y = j; y < min(img.cols, j + sz); y++){
					block(bX, bY) = img(x, y);
					bY++;
				}
				bX++;
				bY = 0;
			}

			block = IDID(block, scale);
			
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
	return result(cropR);
}

int main(){
	Mat_<unsigned char> image;
	image = imread("miniLenna.png", 0);
	//imwrite("B.png", bilinearScale(IDID(image, 5), 5));
	imwrite("C.png", bilinearScale(splitIDID(image, 5), 5));
	imwrite("D.png", splitIDID(image, 5));
	return 0;
}
