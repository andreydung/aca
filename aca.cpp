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

vector<Vec3f> ACA::global_average(const Mat &in, const Mat &lev)
{
	assert(in.type()==21); //CV_32FC3
	assert(lev.type()==4); //CV_32S

	int M=in.rows;
	int N=in.cols;

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

	return average;
}

vector<Vec3f> ACA::local_average(const Mat &in, const Mat &lev, int i, int j, int win_dim)
{
	assert(in.type()==21); //CV_32FC3
	assert(lev.type()==4); //CV_32S
	
	int M=in.rows;
	int N=in.cols;

	vector<float> count(nlevels,0);
	vector<Vec3f> average(nlevels);

	// set boundary coordinates in image
	int a = i-win_dim;
	if(a < 0) 
		a = 0;
	int b = i+win_dim;
	if(b > M-1) 
		b=M-1;
	int c = j-win_dim;
	if(c < 0) 
		c = 0;
	int d = j+win_dim;
	if(d > N-1) 
		d = N-1;

	// check each pixel in range and find local average
	for (int i=a; i < b; i++)
		for (int j=c; j < d; j++)
		{
			int pixlev = lev.at<int>(i,j);
			assert(pixlev<nlevels);

			average[pixlev] += in.at<Vec3f>(i,j);
			count[pixlev]++;
		}

	for (int i=0; i<nlevels; i++)
		average[i] /=count[i];

	return average;
}

int ACA::s_beta(int pixlev, const Mat& lev, int i, int j)
{
	assert(lev.type()==4); //CV_32S
	// Find neighbors, horizontal and vertical direction only
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

	int out =0;
	for (int k=0; k<int(neighbors.size()); k++)
	{
		if (pixlev==neighbors[k])
			out+=1;
		else
			out-=1;
	}
	return out;
}

bool ACA::resegment(const Mat& in, Mat& lev, const vector<Mat>& locav, int i, int j)
{
	bool changed=false;
	int kold = lev.at<int>(i,j);
	float uold, unew;

	uold=energy(kold, in, lev, locav, i, j);
	
	for (int k=0; k<nlevels; k++)
	{
		if (k!=kold)
		{
			unew=energy(k, in, lev, locav, i, j);
			if (unew<uold)
			{
				lev.at<int>(i,j)=k;
				uold=unew;
				changed=true;
			}
		}
	}
	return changed;
}

int ACA::smrf(const Mat& in, Mat& lev, const vector<Mat>& locav)
{
	int M=in.rows;
	int N=in.cols;

	int count=0;
	for (int i=0; i<M; i++)
		for( int j=0; j<N; j++)
		{
			count+=int(resegment(in, lev, locav, i, j));
		}
	return count;
}

float ACA::energy(int pixlev, const Mat &in, const Mat &lev, const vector<Mat> &locav, int i, int j)
{
	float n = float(norm(locav[pixlev].at<Vec3f>(i,j)-in.at<Vec3f>(i,j)));
	return beta*s_beta(pixlev, lev, i,j)+n/ss;
}

vector<Mat> ACA::getlocav(const Mat& in, const Mat& lev, int win_dim)
{
	assert(in.type()==21); //CV_32FC3
	assert(lev.type()==4); //CV_32S
	
	int M = in.rows;
	int N = in.cols;

	vector<Mat> locav;
	vector<Vec3f> averages;

	if (win_dim==0)
	{
		averages=global_average(in, lev);

		for (int k=0; k<nlevels; k++)
		{
			Mat tmp=Mat::zeros(M, N, CV_32FC3);
			for (int i=0; i<M; i++)
				for (int j=0; j<N; j++)
				{
					tmp.at<Vec3f>(i,j)=averages[k];
				}
			locav.push_back(tmp);				
		}
	}
	else
	{
		for (int k=0; k<nlevels; k++)
		{
			Mat tmp=Mat::zeros(M, N, CV_32FC3);
			for (int i=0; i<M; i++)
				for (int j=0; j<N; j++)
				{
					averages=local_average(in, lev, i, j, int(win_dim/2));
					{
						tmp.at<Vec3f>(i,j)=averages[k];
					}
				}
			locav.push_back(tmp);
		}
	}
	return locav;
}

// Mat ACA::aca_seg(const Mat& in)
// {
// 	Mat lev = getkmeans(in);

// 	int M=in.rows;
// 	int N=in.cols;

// 	mcount_mrf=(int)((M+N)*prcnt/2);

// 	if ((M<maxw) && (N<=maxw))	
// 	{
// 		win_dim = 0;	// denotes global averaging for slocavbl
// 		for (int nw = 0; nw < niters_w; nw++)
// 		{
// 			vector<Mat> locav= getlocav(in, lev, win_dim);
// 			for (int n = 0; n < niters_mrf; n++)
// 			{
// 				// calculate re-segmentation
// 				int count_mrf = smrf(in, lev, locav);
// 				printf("   n = %d: %d locations changed\n",n,count_mrf);
// 				if (count_mrf < mcount_mrf) 
// 					break;
// 			}

// 			if (n==0) break;
// 		}

// 		// update window size for local averaging
// 		win_dim_fl = (float)maxw;
// 		win_dim = (int)(win_dim_fl+0.5);
// 		for (int k = 0; k< 30; k++)		// reduce window size by wfactor times until within image dimensions
// 		{
// 			if (win_dim >= hh || win_dim >= ww)
// 			{
// 				win_dim_fl /= wfactor;
// 				win_dim = (int)(win_dim_fl+0.5);
// 			}
// 			else break;
// 		}
// 	}
// 	else	// window size smaller than image size, no global averaging
// 	{
// 		win_dim = maxw;
// 		win_dim_fl = (float)win_dim;
// 	}

// 	for (int m = 0; m < 30; m++)
// 	{
// 		if (win_dim < minw && win_dim < minw) break;

// 		mincount = (int)(win_dim*mult);
// 		if (mincount < 1) mincount = 1;

// 		for (int nw = 0; nw < niters_w; nw++)	// niters_w -> max global averaging iters (default: 10)
// 			{
// 				printf("\nwindow: %d X %d\n",win_dim,win_dim);
// 				vector<Mat> locav= getlocav(in, lev, win_dim);

// 				for (int n = 0; n < niters_mrf; n++)	// niters_mrf -> max re-segmentation iters (default: 30)
// 				{
// 					int count_mrf = smrf(in, lev, locav);
// 					printf("   n = %d: %d locations changed\n",n,count_mrf);
// 					if (count_mrf < mcount_mrf) break;
// 				}

// 				if (n==0) break;
// 			}

// 		win_dim_fl /= wfactor;
// 		win_dim = (int)(win_dim_fl + 0.5);
// 	}

// 	return lev;
// }
