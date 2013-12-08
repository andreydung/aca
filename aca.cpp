#include "aca.h"

int const ACA::NITERS_WINDOW=10;
int const ACA::NITERS_MRF=30;
int const ACA::MAXW = 4096;
int const ACA::MINW = 7;

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

vector<Vec3f> ACA::local_average(const Mat &in, const Mat &lev, int i, int j, int win_dim2)
{
	assert(in.type()==21); //CV_32FC3
	assert(lev.type()==4); //CV_32S
	
	int M=in.rows;
	int N=in.cols;

	vector<float> count(nlevels,0);
	vector<Vec3f> average(nlevels, Vec3f(0,0,0));

	// set boundary coordinates in image
	int a = i-win_dim2;
	if(a < 0) 
		a = 0;
	int b = i+win_dim2;
	if(b > M-1) 
		b=M-1;
	int c = j-win_dim2;
	if(c < 0) 
		c = 0;
	int d = j+win_dim2;
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
	{
		if (count[i]!=0)
			average[i] /=count[i];
	}
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
	cout<<"Here";
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
			int win_dim2 = int(win_dim/2);
			
			//Bilinear interpolation
			for (int i=0; i<M ; i+= win_dim2)
			{
				for (int j=0; j<N ; j+= win_dim2)
				{
					vector<Vec3f> aver=local_average(in, lev, i, j, win_dim2);
					tmp.at<Vec3f>(i,j) = aver[k];
				}
			}

			for (int i=0; i<M; i++)
				for (int j=0; j<N; j++)
				{
					if (j%win_dim2 == 0 && i%win_dim2 == 0) continue;
					tmp.at<Vec3f>(i,j)=bilinear(tmp, win_dim2, win_dim2, i, j);
				}
			locav.push_back(tmp);

		}
	}
	return locav;
}

Vec3f ACA::bilinear(Mat locav, int istep,int jstep,int i,int j)
{
	int M=locav.rows;
	int N=locav.cols;

	int i1,i2,j1,j2;
	Vec3f aa,bb,cc,dd, imgg;
	float alpha,beeta,gamma,delta;

	// get corner positions
	i1 = i-i%istep;
	i2 = i1+istep;
	j1 = j-j%jstep;
	j2 = j1+jstep;

	// get mult. factors
	alpha = (j-j1)/((float) jstep);
	beeta = (i2-i)/((float) istep);
	gamma = (j2-j)/((float) jstep);
	delta = (i-i1)/((float) istep);

	// correct for image boundaries
	if (j2 >= N) j2 = j1;
	if (i2 >= M) i2 = i1;

	// get corner-point values to use for interpolation
	aa = locav.at<Vec3f>(i1,j1);
	bb = locav.at<Vec3f>(i1,j2);
	cc = locav.at<Vec3f>(i2,j2);
	dd = locav.at<Vec3f>(i2,j1);
	
	imgg = dd*delta*gamma;
	imgg += aa*beeta*gamma;
	imgg += bb*alpha*beeta;
	imgg += cc*alpha*delta;
	return imgg;
}

Mat ACA::aca_seg(Mat in)
{
	int M=in.rows;
	int N=in.cols;

	Mat lev = getkmeans(in);
	in.convertTo(in, CV_32F);

	int n;
	int win_dim;
	float win_dim_fl;
	int count_mrf;
	int mcount_mrf=(int)((M+N)*prcnt/2);

	if ((M<MAXW) && (N<=MAXW))	
	{
		
		for (int nw = 0; nw < NITERS_WINDOW; nw++)
		{
			vector<Mat> locav= getlocav(in, lev, 0);
			for (n = 0; n < NITERS_MRF; n++)
			{
				// calculate re-segmentation
				cout<<"Here"<<n<<"\n";
				count_mrf = smrf(in, lev, locav);
				cout<<count_mrf<<" locations changed\n";
				if (count_mrf < mcount_mrf) 
					break;
			}

			if (n==0) break;
		}

		// update window size for local averaging
		win_dim_fl = (float) MAXW;
		win_dim = (int)(win_dim_fl+0.5);

		// reduce window size by wfactor times until within image dimensions
		for (int k = 0; k< 30; k++)		
		{
			if (win_dim >= M || win_dim >= N)
			{
				win_dim_fl /= wfactor;
				win_dim = (int)(win_dim_fl+0.5);
			}
			else break;
		}
	}
	else	// window size smaller than image size, no global averaging
	{
		win_dim = MAXW;
		win_dim_fl = (float)win_dim;
	}

	// for (int m = 0; m < 30; m++)
	// {
	// 	if (win_dim < MINW && win_dim < MINW) break;

	// 	for (int nw = 0; nw < NITERS_WINDOW; nw++)	// niters_w -> max global averaging iters (default: 10)
	// 		{
				
	// 			vector<Mat> locav= getlocav(in, lev, win_dim);

	// 			for (n = 0; n < NITERS_MRF; n++)	// niters_mrf -> max re-segmentation iters (default: 30)
	// 			{
	// 				int count_mrf = smrf(in, lev, locav);
					
	// 				if (count_mrf < mcount_mrf) break;
	// 			}

	// 			if (n==0) break;
	// 		}

	// 	win_dim_fl /= wfactor;
	// 	win_dim = (int)(win_dim_fl + 0.5);
	// }

	return lev;
}
