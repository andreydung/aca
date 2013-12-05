#include "aca.h"

Mat ACA::getkmeans(const Mat &in)
{
	Mat label;
	Mat data=in.reshape(1,in.rows*in.cols);
	data.convertTo(data, CV_32F);
	kmeans(data,nlevels,label,cv::TermCriteria(CV_TERMCRIT_ITER,30,0),1,KMEANS_RANDOM_CENTERS);
	label=label.reshape(1,in.rows);
	return label;
}

Mat ACA::global_average(Mat in, const Mat &lev)
{
	// in: CV_32F
	// lev: CV_32S
	int M=in.rows;
	int N=in.cols;

	in.convertTo(in, CV_32FC3);

	vector<float> count(nlevels,0);
	vector<Vec3f> average(nlevels);

	int pixlev;
	for (int i=0; i < M; i++)
		for (int j=0; j < N; j++)
		{
			pixlev = lev.at<int>(i,j);
			assert (pixlev<nlevels);

			average[pixlev] += in.at<Vec3f>(i,j);
			count[pixlev]++;
		}

	for (int i=0; i<nlevels; i++)
		average[i] /=count[i];


	Mat out=Mat::zeros(M,N,CV_32FC3);
	for (int i=0; i<M; i++)
		for (int j=0; j<N; j++)
		{
			pixlev = lev.at<int>(i,j);
			out.at<Vec3f>(i,j)=average[pixlev];
		}
	return out;
}

vector<int> ACA::neighbor(const Mat& lev, int i, int j)
{
	assert(lev.type()==4); //CV_32S

	vector<int> neighbors;
	int M=lev.rows;
	int N=lev.cols;

	if ((i==0)&&(j==0))
	{
		neighbors.push_back(lev.at<int>(0,1));
		neighbors.push_back(lev.at<int>(1,0));
	}
	else if ((i==0)&&(j==N))
	{
		neighbors.push_back(lev.at<int>(0,N-1));
		neighbors.push_back(lev.at<int>(1,N));
	} 
	else if ((i==M)&&(j==0))
	{
		neighbors.push_back(lev.at<int>(M,1));
		neighbors.push_back(lev.at<int>(M-1,0));
	}
	else if ((i==M)&&(j==N))
	{
		neighbors.push_back(lev.at<int>(M-1,N));
		neighbors.push_back(lev.at<int>(M,N-1));
	}
	else if (i==0)
	{
		neighbors.push_back(lev.at<int>(1,j));
		neighbors.push_back(lev.at<int>(2,j));	
		neighbors.push_back(lev.at<int>(1,j));	
	}
	else if (i==M)
	{
		neighbors.push_back(lev.at<int>(M-1,j));
		neighbors.push_back(lev.at<int>(M,j-1));	
		neighbors.push_back(lev.at<int>(M,j+1));	
	}
	else if (j==0)
	{
		neighbors.push_back(lev.at<int>(i-1,0));
		neighbors.push_back(lev.at<int>(i,1));	
		neighbors.push_back(lev.at<int>(i+1,0));	
	}
	else if (j==N)
	{
		neighbors.push_back(lev.at<int>(i-1,N));
		neighbors.push_back(lev.at<int>(i,N-1));	
		neighbors.push_back(lev.at<int>(i+1,N));	
	}
	else
	{
		neighbors.push_back(lev.at<int>(i-1,j));
		neighbors.push_back(lev.at<int>(i+1,j));
		neighbors.push_back(lev.at<int>(i,j-1));
		neighbors.push_back(lev.at<int>(i,j+1));
		neighbors.push_back(lev.at<int>(i-1,j-1));
		neighbors.push_back(lev.at<int>(i+1,j+1));
		neighbors.push_back(lev.at<int>(i-1,j+1));
		neighbors.push_back(lev.at<int>(i+1,j-1));
	}
	return neighbors;
}