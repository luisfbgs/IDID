#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "interpolations.hpp"

using namespace cv;
using namespace std;

Mat_<uchar> directDownsample(const Mat_<uchar> &img, double scale){
	Mat_<uchar> result;
	resize(img, result, Size(), 1 / scale, 1 / scale, INTER_LINEAR_EXACT);
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

Mat_<short int> IDID(const Mat_<short int> &image, double scale, Itp interpolation = Itp::bilinear){
	int rows = max(1., image.rows / scale);
	int cols = max(1., image.cols / scale);
	Mat_<double> H;
	switch(interpolation){
		default:
			H = bilinearMat(rows, cols, image.rows, image.cols);
	}

	Mat_<double> HT;
	Mat_<double> Y = columnMat<int>(image);
	transpose(H, HT);
	Mat_<double> HHT = HT * H;
	Mat_<double> iv = HHT.inv();
	Mat_<double> res = iv * HT * Y;

	Mat_<short int> newImg(rows, cols);
	int id = 0;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			newImg(i, j) = min(355, (int)round(res(id, 0)));
			id++;
		}
	}
	return newImg.clone();
}

Mat_<short int> splitIDID(const Mat_<short int> &img, double scale){
	int sz = 16;
	Mat_<short int> result(img.rows, img.cols);
	result = 0;
	int lstx, lsty;
	lstx = lsty = 0;
	for(int i = 0; i < img.rows; i += sz){
		int stepx = 0;
		lsty = 0;
		for(int j = 0; j < img.cols; j += sz){
			int rows = min(sz, img.rows - i);
			int cols = min(sz, img.cols - j);
			Mat_<short int> block = Mat_<short int>(rows, cols);

			int bX = 0, bY = 0; 
			for(int x = i; x < i + rows; x++){
				for(int y = j; y < j + cols; y++){
					block(bX, bY) = img(x, y);
					bY++;
				}
				bX++;
				bY = 0;
			}
			block = IDID(block, scale, Itp::bilinear);
			
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

double getPSNR(Mat_<uchar> &image, Mat_<uchar> &res){
	double MSE = 0;
	for(int i = 0; i < res.rows; i++){
		for(int j = 0; j < res.cols; j++){
			double d = res(i, j) - image(i, j);
			MSE += d * d;
		}
	}
	MSE /= res.rows * res.cols;
	return 20 * log10(255) - 10 * log10(MSE);
}

int main(){
	int n;
	scanf("%d", &n);
	for(int i = 0; i < n; i++){
		Mat_<uchar> image;
		char name[50], dir[250];
		scanf("%s", name);
		sprintf(dir, "%s%s", "testimages/", name);
		image = imread(dir, 0);
		
		Mat_<uchar> res = directDownsample(image, 4);
		sprintf(dir, "%s%d%s", "DIRECT/", 4, name);
		imwrite(dir, res);
		sprintf(dir, "%s%d%c%s", "DIRECT/", 4, 'U', name);
		res = bilinearScale(res, 4);
		imwrite(dir, res);
		printf("Direct 4x %s PSRN: %lf\n", name, getPSNR(image, res));

		Mat_<short int> imageInt = image;
		imageInt = imageInt  + 100;
		res = splitIDID(imageInt, 4) - 100;
		sprintf(dir, "%s%d%s", "SIDIDBi/", 4, name);
		imwrite(dir, res);
		sprintf(dir, "%s%d%s%s", "SIDIDBi/", 4, "U", name);
		res = bilinearScale(res, 4);
		sprintf(dir, "%s%d%s%s", "SIDIDBi/", 4, "U", name);
		imwrite(dir, res);
		printf("IDID Bilinear 4x %s PSRN: %lf\n\n", name, getPSNR(image, res));
	}
	return 0;
}
