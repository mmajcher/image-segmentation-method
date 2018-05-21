#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;


void energy_functions_top_level(const Mat_<uchar> &image, const Mat_<double> &kernel, Mat_<double> &energy_1, Mat_<double> &energy_2);
