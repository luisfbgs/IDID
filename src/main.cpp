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
		cerr << name << endl;
		sprintf(dir, "%s%s", "testimages/", name);
		image = imread(dir, 0);
		
		Mat_<uchar> res = directDownsample(image, 2, INTER_NEAREST);
		sprintf(dir, "%s%d%s%s", "DIRECT/", 2, "UDDL", name);
		res = bilinearScale(res, 2);
		imwrite(dir, res);
		printf("Direct 2x DDL %s PSRN: %lf\n", name, getPSNR(image, res));

		res = directDownsample(image, 2, INTER_NEAREST);
		sprintf(dir, "%s%d%s%s", "DIRECT/", 2, "UDDC", name);
		res = bicubicScale(res, 2);
		imwrite(dir, res);
		printf("Direct 2x DDC %s PSRN: %lf\n", name, getPSNR(image, res));

		res = directDownsample(image, 2, INTER_CUBIC);
		sprintf(dir, "%s%d%s%s", "DIRECT/", 2, "UDCL", name);
		res = bilinearScale(res, 2);
		imwrite(dir, res);
		printf("Direct 2x DCL %s PSRN: %lf\n", name, getPSNR(image, res));

		res = directDownsample(image, 2, INTER_CUBIC);
		sprintf(dir, "%s%d%s%s", "DIRECT/", 2, "UDCC", name);
		res = bicubicScale(res, 2);
		imwrite(dir, res);
		printf("Direct 2x DCC %s PSRN: %lf\n", name, getPSNR(image, res));

		Mat_<uchar> resd = splitIDID(image, 2, bilinear);
		sprintf(dir, "%s%d%s%s", "SIDIDBi/", 2, "UILL", name);
		res = bilinearScale(resd, 2);
		imwrite(dir, res);
		printf("IDID Bilinear 2x ILL %s PSRN: %lf\n", name, getPSNR(image, res));

		sprintf(dir, "%s%d%s%s", "SIDIDBi/", 2, "UILC", name);
		res = bicubicScale(resd, 2);
		imwrite(dir, res);
		printf("IDID Bilinear 2x ILC %s PSRN: %lf\n", name, getPSNR(image, res));

		resd = splitIDID(image, 2, bicubic);
		sprintf(dir, "%s%d%s%s", "SIDIDBi/", 2, "UICL", name);
		res = bilinearScale(resd, 2);
		imwrite(dir, res);
		printf("IDID Bicubic 2x ICL %s PSRN: %lf\n", name, getPSNR(image, res));

		sprintf(dir, "%s%d%s%s", "SIDIDBi/", 2, "UICC", name);
		res = bicubicScale(resd, 2);
		imwrite(dir, res);
		printf("IDID Bicubic 2x ICC %s PSRN: %lf\n\n", name, getPSNR(image, res));
	}
	return 0;
}
