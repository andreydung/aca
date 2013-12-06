#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

class ACA
{
public:
	ACA()
	{
		// ACA parameters
		nlevels=4;
		ss=7;
		beta=0.5;
		maxw=4096;
		niters_w=10;
		niters_mrf=30;
		wfactor=2;
	};

	Mat getkmeans(const Mat &in);
	Mat aca_seg(const Mat& in);

	vector<Mat> getlocav(const Mat& in, const Mat& lev, int win_dim);	

	int smrf(const Mat& in, Mat& lev, const vector<Mat>& locav);
	bool resegment(const Mat& in, Mat& lev, const vector<Mat>& locav, int i, int j);
	
private:
	int s_beta(int pixlev, const Mat& lev, int i, int j);
	float energy(int pixlev, const Mat &in, const Mat &lev, const vector<Mat> &locav, int i, int j);

	vector<Vec3f> local_average(const Mat &in, const Mat &lev, int i, int j, int windim);
	vector<Vec3f> global_average(const Mat &in, const Mat &lev);

	int nlevels, maxw;
	int niters_w, niters_mrf;
	float ss;
	float beta;
	int wfactor;
};