#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class ACA()
{
public:
	ACA(int n)
	{
		nlevels=n;
	}
	void getinitkmeans(const Mat &in, Mat &label, int nlevels)
	{
		Mat data=in.reshape(1,in.rows*in.cols);
		data.convertTo(data, CV_32F);
		kmeans(data,nlevels,label,cv::TermCriteria(CV_TERMCRIT_ITER,30,0),1,KMEANS_RANDOM_CENTERS);
		label=label.reshape(1,in.rows);
	};
private:
	int nlevels;	
}