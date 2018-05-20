#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void local_prefitting_functions(const Mat_<uchar> & image, const Mat_<double> & kernel, Mat_<double> &f1, Mat_<double> &f2);

void energy_functions_from_prefiting_functions
(const Mat_<uchar> & image, const Mat_<double> & prefitting_kernel,
 const Mat_<double> & prefit1, const Mat_<double> & prefit2,
 Mat & energy1, Mat & energy2);
