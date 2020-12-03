#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "IDID.hpp"
#include "interpolations.hpp"

using namespace cv;
using namespace std;

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

		res = splitIDID(image, 4);
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
