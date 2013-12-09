#include <UnitTest++/UnitTest++.h>
#include "aca.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}

// TEST(Kmeans)
// {
// 	ACA a(4);
// 	Mat in = imread("fox.jpg",CV_LOAD_IMAGE_GRAYSCALE);
// 	Mat out=a.getkmeans(in);

// 	cout<<type2str(out.type())<<"\n";
// 	cout<<out.rows<<"\n";
// 	cout<<out.cols<<"\n";
// 	cout<<out.channels()<<"\n";
// }

void glb_av(Mat& aver, Mat& lev, Mat& obs, int hh, int ww, int nlevels)
{

	Mat count;	// counts appearances of each level (0,1,...)
	count = Mat::zeros(3,nlevels,CV_32F);

	for (int ii=0;ii < hh;ii++)
	{
		for (int jj=0;jj < ww;jj++)
		{
			// find pixel levels
			int pixlev = lev.at<int>(ii,jj);

			// add pixel values to appropriate counter in aver
			aver.at<float>(0,pixlev) += (float)obs.at<Vec3b>(ii,jj)[0];
			aver.at<float>(1,pixlev) += (float)obs.at<Vec3b>(ii,jj)[1];
			aver.at<float>(2,pixlev) += (float)obs.at<Vec3b>(ii,jj)[2];

			// increment appearance of pixels with level
			count.at<float>(0,pixlev)++;
			count.at<float>(1,pixlev)++;
			count.at<float>(2,pixlev)++;
		}
	}

	// divide each aver bin by its corrseponding count bin
	for (int k = 0; k < nlevels; k++)	// over aver bins
	{
		for (int c = 0; c < 3; c++)		// over each color component
		{
			if (count.at<float>(c,k) < 1) 
				aver.at<float>(c,k) = 1000.0;
			else						  
				aver.at<float>(c,k) /= (count.at<float>(c,k)*255);
		}
	}
}

// LOC_AV: calculates local averages on grid points
void loc_av(Mat& aver,Mat& lev,Mat& obs,int hh,int ww,int nlevels,int win_dim2,int mincount,int i,int j)
{
	Mat count;	// counts appearances of each level (0,1,...)
	count = Mat::zeros(3,nlevels,CV_32F);

	// set boundary coordinates in image
	int ii_i = i-win_dim2;
	if(ii_i < 0) ii_i = 0;
	int ii_f = i+win_dim2;
	if(ii_f > hh-1) ii_f = hh-1;
	int jj_i = j-win_dim2;
	if(jj_i < 0) jj_i = 0;
	int jj_f = j+win_dim2;
	if(jj_f > ww-1) jj_f = ww-1;

	// check each pixel in range and find local average
	for (int ii=ii_i;ii < ii_f;ii++)
	{
		for (int jj=jj_i;jj < jj_f;jj++)
		{
			// find pixel levels
			int pixlev = lev.at<int>(ii,jj);

			// add pixel values to appropriate counter in aver
			aver.at<float>(0,pixlev) += (float)obs.at<Vec3b>(ii,jj)[0];
			aver.at<float>(1,pixlev) += (float)obs.at<Vec3b>(ii,jj)[1];
			aver.at<float>(2,pixlev) += (float)obs.at<Vec3b>(ii,jj)[2];

			// increment appearance of pixels with level
			count.at<float>(0,pixlev)++;
			count.at<float>(1,pixlev)++;
			count.at<float>(2,pixlev)++;
		}
	}

	// divide each aver bin by its corrseponding count bin
	for (int k = 0; k < nlevels; k++)	// over aver bins
	{
		for (int c = 0; c < 3; c++)		// over each color component
		{
			if (count.at<float>(c,k) < mincount) aver.at<float>(c,k) = 1000.0;
			else								 aver.at<float>(c,k) /= (count.at<float>(c,k)*255);
		}
	}
}

void slocavbl(Mat& lev,Mat& obs,Mat& locav,int hh,int ww,int nlevels,int win_dim,int mincount)
{
	//	void loc_av(Mat&,Mat&,Mat&,int,int,int);	// local averaging at grid points
	void loc_av(Mat&,Mat&,Mat&,int,int,int,int,int,int,int);
	void glb_av(Mat&,Mat&,Mat&,int,int,int);	// global averaging over image
	float bilinear(Mat&,int,int,int,int,int,int,int,int);	// bilinear interpolation between grid points

	Mat aver;	// holds average values for each level (0,1,...)
	aver = Mat::zeros(3,nlevels,CV_32F);	// rows: bgr, columns: level (0,1,...)

	if (win_dim == 0)	// global averaging
	{
		// compute global averages
		glb_av(aver,lev,obs,hh,ww,nlevels);

		// store global averages in locav
		for (int i=0;i<hh;i++)
			for (int j=0;j<ww;j++)
				for (int k=0;k<nlevels;k++)
				{
					locav.at<float>(i+k*hh,j+ww*0) = aver.at<float>(0,k);	// b
					locav.at<float>(i+k*hh,j+ww*1) = aver.at<float>(1,k);	// g
					locav.at<float>(i+k*hh,j+ww*2) = aver.at<float>(2,k);	// r
				}
	}
	else	// local averaging
	{
		int win_dim2 = win_dim/2;	// half window size = ...
		int jstep = win_dim2;	// ... stepping length
		int istep = jstep;

		// local averages on grid points
		for (int i=0;i<hh;i += istep)
		{
			for (int j=0;j<ww;j += jstep)
			{
				loc_av(aver,lev,obs,hh,ww,nlevels,win_dim2,mincount,i,j);

				for (int k=0;k<nlevels;k++)
				{
					locav.at<float>(i+k*hh,j+ww*0) = aver.at<float>(0,k);	// b
					locav.at<float>(i+k*hh,j+ww*1) = aver.at<float>(1,k);	// g
					locav.at<float>(i+k*hh,j+ww*2) = aver.at<float>(2,k);	// r
				}
			}
		}

		// bilinear interpolation
		for (int i=0;i<hh;i++)
		{
			for (int j=0;j<ww;j++)
			{
				if (j%jstep == 0 && i%istep == 0) continue;

				for (int k=0;k<nlevels;k++)
				{
					for (int c=0;c<3;c++)
					{
						locav.at<float>(i+k*hh,j+ww*c) = bilinear(locav,k,c,istep,jstep,hh,ww,i,j);
					}
				}
			}
		}
	}
}


// BILINEAR: computes inner pixel averages using grid points averages
float bilinear(Mat& locav,int k,int c,int istep,int jstep,int hh,int ww,int i,int j)
{
	int i1,i2,j1,j2;
	float aa,bb,cc,dd;
	float imgg,alpha,beeta,gamma,delta;

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
	if (j2 >= ww) j2 = j1;
	if (i2 >= hh) i2 = i1;

	// get corner-point values to use for interpolation
	aa = locav.at<float>(i1+k*hh,j1+ww*c);
	bb = locav.at<float>(i1+k*hh,j2+ww*c);
	cc = locav.at<float>(i2+k*hh,j2+ww*c);
	dd = locav.at<float>(i2+k*hh,j1+ww*c);

	// determine value to return
	if (aa < 1000 && bb < 1000 && cc < 1000 && dd < 1000)
	{
		imgg = dd*delta*gamma;
		imgg += aa*beeta*gamma;
		imgg += bb*alpha*beeta;
		imgg += cc*alpha*delta;
		return imgg;
	}
	if (aa+bb+cc+dd >= 1000) return 1000;
	if (aa+bb+cc+dd >= 3000)
	{
		if (aa < 1000 && (alpha <= .5 && delta <= .5)) return aa;
		if (bb < 1000 && (gamma <= .5 && delta <= .5)) return bb;
		if (cc < 1000 && (beeta <= .5 && gamma <= .5)) return cc;
		if (dd < 1000 && (alpha <= .5 && beeta <= .5)) return dd;
		else return 1000;
	}
	if (aa+bb+cc+dd >= 2000)
	{
		if (aa == 1000 && bb == 1000 && beeta <= .5)
			return (cc*alpha+dd*gamma);
		if (aa == 1000 && dd == 1000 && gamma <= .5)
			return (bb*beeta+cc*delta);
		if (bb == 1000 && cc == 1000 && alpha <= .5)
			return (aa*beeta+dd*delta);
		if (cc == 1000 && dd == 1000 && delta <= .5)
			return (aa*gamma+bb*alpha);
		if (aa == 1000 && cc == 1000 && alpha <= .5 && beeta <= .5)
			return ((bb*alpha*beeta+dd*delta*gamma)/(alpha*beeta+delta*gamma));
		if (aa == 1000 && cc == 1000 && gamma <= .5 && delta <= .5)
			return ((bb*alpha*beeta+dd*delta*gamma)/(alpha*beeta+delta*gamma));
		if (bb == 1000 && dd == 1000 && delta <= .5 && alpha <= .5)
			return ((aa*beeta*gamma+cc*alpha*delta)/(beeta*gamma+alpha*delta));
		if (bb == 1000 && dd == 1000 && beeta <= .5 && gamma <= .5)
			return ((aa*beeta*gamma+cc*alpha*delta)/(beeta*gamma+alpha*delta));
		else return 1000;
	}
	if (aa == 1000 && (alpha >= .5 || delta >= .5))
	{
		imgg = dd*delta*gamma;
		imgg += bb*alpha*beeta;
		imgg += cc*alpha*delta;
		return imgg/(delta*gamma+alpha*beeta+alpha*delta);
	}
	if (bb == 1000 && (gamma >= .5 || delta >= .5))
	{
		imgg = dd*delta*gamma;
		imgg += aa*beeta*gamma;
		imgg += cc*alpha*delta;
		return imgg/(delta*gamma+alpha*delta+beeta*gamma);
	}
	if (cc == 1000 && (gamma >= .5 || beeta >= .5))
	{
		imgg = dd*delta*gamma;
		imgg += aa*beeta*gamma;
		imgg += bb*alpha*beeta;
		return imgg/(delta*gamma+alpha*beeta+beeta*gamma);
	}
	if (dd == 1000 && (alpha >= .5 || beeta >= .5))
	{
		imgg = aa*beeta*gamma;
		imgg += bb*alpha*beeta;
		imgg += cc*alpha*delta;
		return imgg/(alpha*beeta+alpha*delta+beeta*gamma);
	}

	return 1000;
}


// TEST(GLOBAL_AVERAGE)
// {
// 	ACA a;
// 	Mat im = imread("fox.jpg");
// 	int M=im.rows;
// 	int N=im.cols;
// 	int nlevels=4;

// 	Mat lev = a.getkmeans(im);
// 	Mat locav = Mat::zeros(M*nlevels, N*3,CV_32F);	
// 	slocavbl(lev, im, locav, im.rows, im.cols, 4, 0, 0);

// 	im.convertTo(im, CV_32F);	
// 	vector<Mat> mylocav=a.getlocav(im, lev, 0);
	
// 	for (int i=0; i<nlevels; i++)
// 	{
// 		vector<Mat> channels(3);
// 		split(mylocav[i], channels);

// 		for (int j=0; j<3; j++)
// 		{
// 			Mat oldmat = locav(Rect(j*N, i*M, N, M))*255;
// 			Mat newmat=channels[j];

// 			for (int k=0; k<M; k++)
// 				for( int m=0; m<N; m++)
// 					CHECK_EQUAL(0,int(oldmat.at<float>(k,m)-newmat.at<float>(k,m)));
// 		}
// 	}
// }

// TEST(LOCAL_AVERAGE)
// {
// 	ACA a;
// 	Mat im = imread("fox.jpg");
// 	int M=im.rows;
// 	int N=im.cols;
// 	int nlevels=4;

// 	int win_dim=128;
// 	int win_dim2=64;

// 	Mat lev = a.getkmeans(im);
// 	Mat locav = Mat::zeros(M*nlevels, N*3,CV_32F);	
// 	slocavbl(lev, im, locav, im.rows, im.cols, 4, win_dim, 0);

// 	im.convertTo(im, CV_32F);	
// 	vector<Mat> mylocav=a.getlocav(im, lev, win_dim);
	
// 	for (int i=0; i<nlevels; i++)
// 	{
// 		vector<Mat> channels(3);
// 		split(mylocav[i], channels);

// 		for (int j=0; j<3; j++)
// 		{
// 			Mat oldmat = locav(Rect(j*N, i*M, N, M));
// 			Mat newmat=channels[j];

// 			for (int k=0; k<M; k++)
// 				for( int m=0; m<N; m++)
// 				{
// 					float tmp = oldmat.at<float>(k,m);
// 					if (tmp<10000)
// 					{
// 						cout<<255*tmp<<" "<<newmat.at<float>(k,m)<<"\n";
// 						CHECK_EQUAL(0,int(255*tmp-newmat.at<float>(k,m)));
// 					}
// 				}
// 		}
// 	}
// }

TEST(ACA_SEG)
{
	ACA a;
	Mat im=imread("1.tif");

	im.convertTo(im, CV_32F);
	Mat lev=a.getkmeans(im);
	vector<Mat> locav=a.getlocav(im, lev, 0);
	
	Mat out=a.aca_seg(im);
	imwrite("output.png", out*40);
}


int main()
{
		return UnitTest::RunAllTests();
}

