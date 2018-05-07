#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <limits>

#include "src/active_contour_method.h"
#include "src/local_prefitting_functions.h"
#include "src/draw_contour_from_acm_matrix.h"

using namespace cv;
using namespace std;

int main() {

    Mat image = imread("examples/2.bmp", IMREAD_GRAYSCALE);


    // INITIAL LSF

    Mat initialLSF = Mat::ones(image.size(), CV_32FC1);

    Rect random_rectangle(Point(40, 15), Size(20, 20));

    initialLSF(random_rectangle).setTo(-1.0);

    // -- or only ones? (or zeros?)
    // initialLSF = Mat::ones(image.size(), CV_32FC1);


    // PARAMETERS

    int mu = 1;
    double nu = 0.01 * 255 * 255;
    int lambda1 = 1;
    int lambda2 = lambda1;
    double epsilon = 1.0;
    double timestep = 0.02;
    int iterations_number = 200;
    double sigma = 2; // control local size


    // LOCAL PRE-FITTING FUNCTIONS

    int gauss_kernel_size = round(2 * sigma) * 2 + 1;
    Mat gauss_kernel_1d = getGaussianKernel(gauss_kernel_size, sigma, CV_32FC1);
    Mat gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t();

    Mat prefitting_kernel = gauss_kernel_2d;


    Mat prefitting_1;
    Mat prefitting_2;

    local_prefitting_functions(image, prefitting_kernel, prefitting_1, prefitting_2);


    // CALCULATE ENERGY FUNCTIONS

    Mat energy1, energy2;

    energy_functions_from_prefiting_functions(image, prefitting_kernel, prefitting_1, prefitting_2, energy1, energy2);


    // LEVEL SET EVOLUTION

    Mat LSF = initialLSF.clone();

    namedWindow("x", WINDOW_NORMAL);
    resizeWindow("x", 500, 500);

    for(int i=0; i<iterations_number; i++) {
      LSF = active_contour_with_local_prefitting_functions(LSF, nu, timestep, mu, epsilon, lambda1, lambda2, energy1, energy2);

      // print every X iteration

      int X = 10;

      if(i % X == 0) {
        Mat new_image = decorate_with_contours_from_acm_matrix(image, LSF);

        resize(new_image, new_image, Size(250,250));

	// text: iteration number X
        putText(new_image, "iter: "+to_string(i), Point(20,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250));

        imshow("x", new_image);

        waitKey(100);
      }

    }

    waitKey();


    return 0;
}
