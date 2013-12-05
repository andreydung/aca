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

// TEST(global_average)
// {
// 	ACA a(4);
// 	Mat im = imread("fox.jpg",CV_LOAD_IMAGE_COLOR);
	
//   Mat lev=a.getkmeans(im);
//   Mat out=a.global_average(im, lev);
//   out.convertTo(out,CV_8U);
// 	imshow("global", out);
// 	waitKey(0);
// }

TEST(neighbor)
{
  ACA a;
  Mat in = imread("fox.jpg",CV_LOAD_IMAGE_GRAYSCALE);
  Mat lev = a.getkmeans(in);
  cout<<"Type "<< lev.type()<<"\n";
  vector<int> n=a.neighbor(in,0,0);
  for (int i=0; i < n.size(); i++)
    cout<<n[i]<<" ";
  cout<<" ";
}

int main()
{
    return UnitTest::RunAllTests();
}

