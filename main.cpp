/*
 * picacalgc: An Adaptive Clustering Algorithm For Image Segmentation
 *			  4 clique types (2-pixel cliques);
 *			  RGB images only;
 *	
 *		input image: original RGB image
 *		
 * * *  optional arguments
 *
 *		   beta: Gibbs parameter
 *			std: standard deviation of the noise
 *		wfactor: window reduction factor
 *		   minw: minimum window size
 *		   maxw: maximum window size
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#define MAXRS 4096
#define NLEVELS 16
#define UINT8 int

using namespace cv;
using namespace std;

// GETINITKMEANS: calculate initial kmeans segmentation (lev,img) //
void getinitkmeans(const Mat &in, Mat &label, int nlevels)
{
	Mat data=in.reshape(1,in.rows*in.cols);
	data.convertTo(data, CV_32F);
	kmeans(data,nlevels,label,cv::TermCriteria(CV_TERMCRIT_ITER,30,0),1,KMEANS_RANDOM_CENTERS);
	label=label.reshape(1,in.rows);
}

// Main program //
int main(int argc, char* argv[])
{
	float	std0, std1, std2, prcnt, mult, win_dim_fl, wfactor, beta;
	int		nw, niters_w, n, niters_mrf;
	int		mcount_mrf, mincount;
	int		win_dim, minw, maxw;
	int		nlevels;
	Mat obs;		// original image (size: dim) (type: uchar, 3 channels)
	Mat lev;		// kmeans levels {0,1,2,...} (size: dim) (type: uchar, 3 same-value channels)
	Mat in;		// segmentation {0,...,255} (size: dim) (type: uchar, 3 same-value channels)
	Mat locav;		// local average values (size: (dim*nlevels)x3) (type: float)
	Mat seg;		// ACA-segmented image
	string infile  = "fox.jpg";
	string outfile = "out.jpg";	// output filename location

	// set (default) parameters //
	beta = 0.5;			// default beta
	std0 = 7.0;			// default std0
	std1 = 7.0;			// default std1
	std2 = 7.0;			// default std2
	niters_w = 10;		// default maximum number of window iters
	niters_mrf = 30;	// default maximum number of mrf iters
	prcnt = 0.1;		// mcount_mrf = 0.5*(hh+ww)*prcnt
	mult = 1.0;			// mincount = (cw2+rw2+1)*mult
	wfactor = 2;		// default window reduction factor 
	maxw = MAXRS;		// default maximum window size
	minw = 7;			// default minimum window size
	nlevels = 4;		// detaulf number of kmeans clusters

	
	// read input image (obs) //
	in = imread(infile,CV_LOAD_IMAGE_GRAYSCALE);
	getinitkmeans(in,lev,4);

	// initialize segmented image
	seg = Mat::zeros(hh,ww,CV_8UC3);

	// global averaging //

	printf("\n---Global--\n");

	locav = Mat::zeros(hh*nlevels,ww*3,CV_32F);	// conceptually, 4x3 grid of sub-matrices of size hh*ww:
												// locav{level 0,b} locav{level 0,g} locav{level 0,r}
												// locav{level 1,b} locav{level 1,g} locav{level 1,r}
												// ...
												// access: locav.at<float>(i+k*hh,j+c*ww)
	
	mcount_mrf = (int)(.5*(hh+ww)*prcnt);	// stopping criteria

	if (maxw >= hh && maxw >= ww)	// each dimension is less than maxw (default: 4096)
	{
		win_dim = 0;	// denotes global averaging for slocavbl

		for (nw = 0; nw < niters_w; nw++)	// niters_w -> max global averaging iters (default: 10)
		{
			printf("\nwindow: global average\n");
				
			// calculate local averages
			slocavbl(lev,obs,locav,hh,ww,nlevels,win_dim,0);

			for (n = 0; n < niters_mrf; n++)	// niters_mrf -> max re-segmentation iters (default: 30)
			{
				// calculate re-segmentation
				int count_mrf = smrf3(lev,obs,locav,hh,ww,nlevels,std0,std1,std2,beta);

				printf("   n = %d: %d locations changed\n",n,count_mrf);

				if (count_mrf < mcount_mrf) break;
			}

			if (n==0) break;
		}

		// update window size for local averaging
		win_dim_fl = (float)maxw;
		win_dim = (int)(win_dim_fl+0.5);
		for (int k = 0; k< 30; k++)		// reduce window size by wfactor times until within image dimensions
		{
			if (win_dim >= hh || win_dim >= ww)
			{
				win_dim_fl /= wfactor;
				win_dim = (int)(win_dim_fl+0.5);
			}
			else break;
		}
	}
	else	// window size smaller than image size, no global averaging
	{
		win_dim = maxw;
		win_dim_fl = (float)win_dim;
	}


	// local averaging //

	printf("\n--Local--\n");

	for (int m = 0; m < 30; m++)
	{
		setlevels(lev,img,levels,hh,ww);

		// construct segmented image
		getSeg(seg,lev,locav,hh,ww,nlevels);
		
		// show image
		imshow("fox_kmeans",seg);
		waitKey(1);

		// write out current img
		imwrite(outfile,seg);

		if (win_dim < minw && win_dim < minw) break;

		mincount = (int)(win_dim*mult);
		if (mincount < 1) mincount = 1;

		for (nw = 0; nw < niters_w; nw++)	// niters_w -> max global averaging iters (default: 10)
			{
				printf("\nwindow: %d X %d\n",win_dim,win_dim);
				
				slocavbl(lev,obs,locav,hh,ww,nlevels,win_dim,mincount);

				for (n = 0; n < niters_mrf; n++)	// niters_mrf -> max re-segmentation iters (default: 30)
				{
					int count_mrf = smrf3(lev,obs,locav,hh,ww,nlevels,std0,std1,std2,beta);

					printf("   n = %d: %d locations changed\n",n,count_mrf);

					if (count_mrf < mcount_mrf) break;
				}

				if (n==0) break;
			}

		win_dim_fl /= wfactor;
		win_dim = (int)(win_dim_fl + 0.5);
	}

	setlevels(lev,img,levels,hh,ww);

	// write final segmentation to file (img) //
	imwrite(outfile,seg);
	waitKey(1);

	// indicate end of execution
	printf("\n--End--\n");
	waitKey(0);

	// check if file was written successfully
	if (...)
	{ printf("Could not save\n"); }
	else {printf("Saved properly\n");}

	return 0;
}


// SETLEVELS: write updated lev info (0,85,...,255) to img //
void setlevels(Mat& lev,Mat& img,vector<UINT8>& levels,int hh,int ww)
{
	for (int i=0;i<hh;i++)
	{
		for (int j=0;j<ww;j++)
		{
			img.at<UINT8>(i,j) = levels[lev.at<UINT8>(i,j)];
		}
	}
}

// GETSEG: updates segmented image from local averages and lev information
void getSeg(Mat& seg,Mat& lev,Mat& locav,int hh,int ww,int nlevels)
{
	for (int i=0;i<hh;i++)
	{
		for (int j=0;j<ww;j++)
		{
			int pix_lev = lev.at<UINT8>(i,j);
			for (int c=0;c<3;c++)
			{
				seg.at<Vec3b>(i,j)[c] = (uchar)(locav.at<float>(i+pix_lev*hh,j+ww*c)*255);
			}
		}
	}
}
/* SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL SLOCAVBL */

// SLOCAVBL: calculate local averages by sliding window and interpolation
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

// GLB_AV: calculates global averages on four grid points
void glb_av(Mat& aver, const Mat& lev, const Mat& obs, int nlevels)
{
	int hh=aver.rows;
	int ww=aver.cols;

	Mat count;	// counts appearances of each level (0,1,...)
	count = Mat::zeros(3,nlevels,CV_32F);

	for (int ii=0;ii < hh;ii++)
	{
		for (int jj=0;jj < ww;jj++)
		{
			// find pixel levels
			int pixlev = lev.at<UINT8>(ii,jj);

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
			int pixlev = lev.at<UINT8>(ii,jj);

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

/* SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 SMRF3 */
// SMRF3: calculate re-segmentation using markov random field model and MAP probability minimization
int smrf3(Mat& lev,Mat& obs,Mat& locav,int hh,int ww,int nlevels,float std0,float std1,float std2,float beta)
{
	int resegmentMAP(Mat&,Mat&,Mat&,int,int,int,float,float,float,float,int,int);
	float s_beta(int,Mat&,int,int,int,int,float);
	float s_beta_bd(int,Mat&,int,int,int,int,float);

	float sss0 = 2*std0*std0;	// variance of Guassian noise in signal model (2*sigma^2)
	float sss1 = 2*std1*std1;
	float sss2 = 2*std2*std2;
	
	int count = 0;	// number of pixels changed (accumulator)
	
	// re-segment each pixel one at a time
	for (int i = 0; i < hh; i++)
		for (int j = 0; j < ww; j++)
		{
			count += resegmentMAP(lev,obs,locav,hh,ww,nlevels,sss0,sss1,sss2,beta,i,j);
		}

	return count;
}

// RESEGMENTMAP: calculate MAP prob and resegment a single pixel
int resegmentMAP(Mat& lev,Mat& obs,Mat& locav,int hh,int ww,int nlevels,float sss0,float sss1,float sss2,float beta,int i,int j)
{
	float s_beta(int,Mat&,int,int,int,int,float);
	float s_beta_bd(int,Mat&,int,int,int,int,float);

	int kold, knew, ch_flag, bound_pix;
	float uold, unew, dif;

	// determine if boundary pixel
	if ( (i == 0) || (i == hh-1) || (j == 0) || (j == ww-1) )
		bound_pix = 1;
	else bound_pix = 0;

	// calculate current MAP prob //
	// calculate clique potentials
	kold = lev.at<UINT8>(i,j);
	
	if (bound_pix)
		uold = s_beta_bd(kold,lev,i,j,hh,ww,beta);
	else uold = s_beta(kold,lev,i,j,hh,ww,beta);

	// calculate closeness to original image
	dif = ((float)obs.at<Vec3b>(i,j)[0])/255 - locav.at<float>(i+kold*hh,j+ww*0);
	uold += (dif*dif)/sss0;
	dif = ((float)obs.at<Vec3b>(i,j)[1])/255 - locav.at<float>(i+kold*hh,j+ww*1);
	uold += (dif*dif)/sss1;
	dif = ((float)obs.at<Vec3b>(i,j)[2])/255 - locav.at<float>(i+kold*hh,j+ww*2);
	uold += (dif*dif)/sss2;

	// calculate MAP prob at each other level, check for improvement
	ch_flag = 0;

	for (int k = 0; k < nlevels; k++)
	{
		if (k != kold)
		{
			// calculate clique potentials
			knew = k;

			if (bound_pix)
				unew = s_beta_bd(knew,lev,i,j,hh,ww,beta);
			else unew = s_beta(knew,lev,i,j,hh,ww,beta);

			// calculate closeness to original image
			dif = ((float)obs.at<Vec3b>(i,j)[0])/255 - locav.at<float>(i+knew*hh,j+ww*0);
			unew += (dif*dif)/sss0;
			dif = ((float)obs.at<Vec3b>(i,j)[1])/255 - locav.at<float>(i+knew*hh,j+ww*1);
			unew += (dif*dif)/sss1;
			dif = ((float)obs.at<Vec3b>(i,j)[2])/255 - locav.at<float>(i+knew*hh,j+ww*2);
			unew += (dif*dif)/sss2;

			if (unew < uold) /* downhill move */
			{
				lev.at<UINT8>(i,j) = knew;
				uold = unew;
				ch_flag = 1;
			}
		}
	}

	return ch_flag;
}

// S_BETA: calculate clique potentials in 8-neighborhood for inner pixel
float s_beta(int pix_lev,Mat& lev,int i,int j,int hh,int ww,float beta)
{
	/*		p b c
	 *		h a d
	 *		g f e
	 */

	float uu = 0;

	// calculate neighboring pixel levels
	int b,c,d,e,f,g,h,p;

	b = lev.at<UINT8>(i-1,j);
	c = lev.at<UINT8>(i-1,j+1);
	d = lev.at<UINT8>(i,j+1);
	e = lev.at<UINT8>(i+1,j+1);
	f = lev.at<UINT8>(i+1,j);
	g = lev.at<UINT8>(i+1,j-1);
	h = lev.at<UINT8>(i,j-1);
	p = lev.at<UINT8>(i-1,j-1);

	// accumulate clique potentials
	if (pix_lev == b) uu -= beta;
	else			  uu += beta;
	if (pix_lev == c) uu -= beta;
	else			  uu += beta;
	if (pix_lev == d) uu -= beta;
	else			  uu += beta;
	if (pix_lev == e) uu -= beta;
	else			  uu += beta;
	if (pix_lev == f) uu -= beta;
	else			  uu += beta;
	if (pix_lev == g) uu -= beta;
	else			  uu += beta;
	if (pix_lev == h) uu -= beta;
	else			  uu += beta;
	if (pix_lev == p) uu -= beta;
	else			  uu += beta;

	return uu;
}

// S_BETA_BD: calculate clique potentials in 8-neighborhood for boundary pixel
float s_beta_bd(int pix_lev,Mat& lev,int i,int j,int hh,int ww,float beta)
{
	float uu = 0.0;

	// accumulate clique potentials
	if (j > 0)	// check pixel to left
	{
		if (pix_lev == lev.at<UINT8>(i,j-1)) uu -= beta;	// all three lev channels have the same value
		else										 uu += beta;
	}
	if (j < ww-1)	// check pixel to right
	{
		if (pix_lev == lev.at<UINT8>(i,j+1)) uu -= beta;
		else										 uu += beta;
	}
	// check pixels above
	if (i > 0)
	{
		// above
		if (pix_lev == lev.at<UINT8>(i-1,j)) uu -= beta;
		else										 uu += beta;
		if (j > 0)	// above-left
		{
			if (pix_lev == lev.at<UINT8>(i-1,j-1)) uu -= beta;
			else										   uu += beta;
		}
		if (j < ww-1)	// above-right
		{
			if (pix_lev == lev.at<UINT8>(i-1,j+1)) uu -= beta;
			else										   uu += beta;
		}
	}
	// check pixels below
	if (i < hh-1)
	{
		// below
		if (pix_lev == lev.at<UINT8>(i+1,j)) uu -= beta;
		else										 uu += beta;
		if (j > 0)	// below-left
		{
			if (pix_lev == lev.at<UINT8>(i+1,j-1)) uu -= beta;
			else										   uu += beta;
		}
		if (j < ww-1)	// below-right
		{
			if (pix_lev == lev.at<UINT8>(i+1,j+1)) uu -= beta;
			else										   uu += beta;
		}
	}

	return uu;
}




