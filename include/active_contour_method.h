#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;


Mat_<double> acm_advance
(Mat_<double> LSF_init, double nu, double timestep, double mu, double epsilon, double lambda1, double lambda2, Mat_<double> energy1, Mat_<double> energy2);

vector<vector<Point>> acm_get_contours(const Mat_<double> &LSF);

Mat_<uchar> decorate_with_contours_from_acm_matrix
(const Mat_<uchar> &image, const Mat_<double> &LSF);
