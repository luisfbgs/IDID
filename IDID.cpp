#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

Mat_<unsigned char> directDownsample(const Mat_<unsigned char> &img, double scale){
	Mat_<unsigned char> result;
	resize(img, result, Size(), 1 / scale, 1 / scale, INTER_NEAREST);
	return result;
}

Mat_<double> bilinearMat(int rows, int cols, double scale){
	int nrows = int(scale * rows), ncols = int(scale * cols);
	Mat_<double> H(nrows * ncols, rows * cols);
	H *= 0;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			int x = (.5 + i) * scale + .5, y = (.5 + j) * scale + .5;
			int nx = (.5 + i + 1) * scale + .5, ny = (.5 + j + 1) * scale + .5;
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
	return H;
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

Mat_<unsigned char> bilinearIDID(const Mat_<unsigned char> &image, double scale){
	Mat_<unsigned char> img = directDownsample(image, scale);
	int rows = img.rows;
	int cols = img.cols;
	Mat_<double> H = bilinearMat(rows, cols, scale);

	Mat_<double> HT;
	Mat_<double> Y = columnMat<unsigned char>(image);
	transpose(H, HT);
	Mat_<double> HHT = HT * H;
	Mat_<double> iv = HHT.inv();
	iv = iv * HT;
	Mat_<double> res = iv * Y;

	Mat_<unsigned char> newImg(img.rows, img.cols);
	int id = 0;
	for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			newImg(i, j) = min(255, (int)round(res(id++, 0)));
		}
	}
	return newImg;
}

Mat_<unsigned char> bilinearScale(const Mat_<unsigned char> &img, double scale){
	Mat_<unsigned char> re;
	resize(img, re, Size(), scale, scale, INTER_LINEAR_EXACT);
	return re;
	int rows = img.rows;
	int cols = img.cols;
	Mat_<unsigned char> result(int(scale * rows), int(scale * cols));
	int nrows = result.rows, ncols = result.cols;
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
	return result;
}

int main(){
	Mat_<unsigned char> image;
	image = imread("miniLenna.png", 0);
	imwrite("B.png", bilinearScale(bilinearIDID(image, 4), 4));
	return 0;
}
