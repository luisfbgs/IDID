#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

Mat_<unsigned char> bilinearScale(Mat_<unsigned char> img, double scale){
	int rows = img.rows;
	int cols = img.cols;
	Mat_<unsigned char> result(int(scale * rows), int(scale * cols));
	int nrows = result.rows, ncols = result.cols;
	printf("%d %d\n", nrows, ncols);
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			int x = i * scale, y = j * scale;
			int nx = (i + 1) * scale, ny = (j + 1) * scale;
			int ni = min(i+1, rows-1), nj = min(j+1, cols-1);
			for(int x1 = x; x1 < min(nrows, nx); x1++){
				for(int y1 = y; y1 < min(ncols, ny); y1++){
					result(x1, y1) = round(
							double(ny - y1) / (ny - y) * (nx - x1) / (nx - x) * img(i, j) +
						   	double(ny - y1) / (ny - y) * (x1 - x) / (nx - x) * img(ni, j) +
						   	double(y1 - y) / (ny - y) * (nx - x1) / (nx - x) * img(i, nj) +
						   	double(y1 - y) / (ny - y) * (x1 - x) / (nx - x) * img(ni, nj));
				}
			}
		}
	}
	return result;
}

int main(){
	Mat_<unsigned char> image;
	image = imread("lenna.png", 0);
	printf("%d\n", image(0, 0));
	imwrite("D.png", bilinearScale(image, 3));
	return 0;
}
