#include "aca.h"

using namespace cv;

int main(int argc, char* argv[])
{
	if (argc!=2)
	{
		cout<<"Usage: aca inputimage\n";
		return 1;
	}
	Mat im;
	im = imread(argv[1]);

	if (!im.data)
	{
		cout<<"Invalid image\n";
		return 1;
	}

	ACA a;
	Mat out=a.aca_seg(im);
	imwrite("out.png", out*30);
	return 0;
}