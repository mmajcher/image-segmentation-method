#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;


Mat active_contour_with_local_prefitting_functions
(Mat LSF, double nu, double timestep, double mu, double epsilon, double lambda1, double lambda2, Mat energy1, Mat energy2);
