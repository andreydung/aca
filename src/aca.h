#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

class ACA
{
public:
	static int const NITERS_WINDOW;
	static int const NITERS_MRF;
	static int const MAXW;
	static int const MINW;

	ACA()
	{
		// ACA parameters
		nlevels=4;
		ss=7;
		beta=0.5;
		wfactor=2;
		prcnt=0.1;
	};

	Mat getkmeans(const Mat &in);
	vector<Mat> getlocav(const Mat& in, const Mat& lev, int win_dim);	
	Mat aca_seg(Mat in);
	
private:
	int smrf(const Mat& in, Mat& lev, const vector<Mat>& locav);
	bool resegment(const Mat& in, Mat& lev, const vector<Mat>& locav, int i, int j);
	Vec3f bilinear(Mat locav, int istep,int jstep,int i,int j);

	int s_beta(int pixlev, const Mat& lev, int i, int j);
	float energy(int pixlev, const Mat &in, const Mat &lev, const vector<Mat> &locav, int i, int j);

	vector<Vec3f> local_average(const Mat &in, const Mat &lev, int i, int j, int windim);
	vector<Vec3f> global_average(const Mat &in, const Mat &lev);

	int nlevels;
	float ss;
	float beta, prcnt;
	int wfactor;
};