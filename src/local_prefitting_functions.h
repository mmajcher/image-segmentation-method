#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void local_prefitting_functions(const Mat_<uchar> & image, const Mat & kernel, Mat &f1, Mat &f2);
void energy_functions_from_prefiting_functions(const Mat_<uchar> & image, const Mat & prefitting_kernel, const Mat & prefit1, const Mat & prefit2, Mat & energy1, Mat & energy2);
