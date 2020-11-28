#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

Mat_<unsigned char> bilinearScale(Mat_<unsigned char> img, double scale){
	int rows = img.rows;
	int cols = img.cols;
	int nrows = int(scale * rows), ncols = int(scale * cols);
	Mat_<double> H(nrows * ncols, rows * cols);
	H *= 0;
	Mat_<double> aux(rows * cols, 1);
	int id = 0;
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			aux(id++, 0) = img(i, j);
			int x = i * scale, y = j * scale;
			int nx = (i + 1) * scale, ny = (j + 1) * scale;
			int ni = min(i+1, rows-1), nj = min(j+1, cols-1);
			for(int x1 = x; x1 < min(nrows, nx); x1++){
				for(int y1 = y; y1 < min(ncols, ny); y1++){
					int k = x1 * ncols + y1;
					H(k, i * cols + j) += double(ny - y1) / (ny - y) * (nx - x1) / (nx - x);
					H(k, ni * cols + j) += double(ny - y1) / (ny - y) * (x1 - x) / (nx - x);
					H(k, i * cols + nj) += double(y1 - y) / (ny - y) * (nx - x1) / (nx - x);
					H(k, ni * cols + nj) += double(y1 - y) / (ny - y) * (x1 - x) / (nx - x);
				}
			}
		}
	}
	Mat_<double> mult = H * aux;
	Mat_<unsigned char> result(nrows, ncols);
	id = 0;
	for(int i = 0; i < nrows; i++){
		for(int j = 0; j < ncols; j++){
			result(i, j) = round(mult(id++, 0));
		}
	}
	return result;
}

int main(){
	Mat_<unsigned char> image;
	image = imread("D.png", 0);
	printf("%d\n", image(0, 0));
	imwrite("F.png", bilinearScale(image, 2));
	return 0;
}
