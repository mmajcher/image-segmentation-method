#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

Mat decorate_with_contours_from_acm_matrix(const Mat& image, const Mat& LSF);