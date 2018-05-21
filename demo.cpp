#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <limits>

#include "active_contour_method.h"
#include "local_prefitting_functions.h"

using namespace cv;
using namespace std;



int main() {

    Mat_<uchar> image = imread("images/2.bmp", IMREAD_GRAYSCALE);


    // INITIAL LSF

    Mat_<double> initialLSF = Mat::ones(image.size(), CV_64FC1);

    Rect some_rectangle(Point(40, 15), Size(20, 20));

    initialLSF(some_rectangle).setTo(-1.0);

    // -- or only ones? (or zeros?)
    // initialLSF = Mat::ones(image.size(), CV_64FC1);


    // PARAMETERS

    int mu = 1;
    double nu = 0.01 * 255 * 255;
    int lambda1 = 1;
    int lambda2 = lambda1;
    double epsilon = 1.0;
    double timestep = 0.02;
    int iterations_number = 200;
    double sigma = 2; // control local size


    // PREPARE ENERGY FUNCTIONS (based on LOCAL PREFITTING FUNCTIONS)

    int gauss_kernel_size = round(2 * sigma) * 2 + 1;
    Mat_<double> gauss_kernel_1d = getGaussianKernel(gauss_kernel_size, sigma, CV_64FC1);
    Mat_<double> gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t();

    Mat_<double> prefitting_kernel = gauss_kernel_2d;
    Mat_<double> energy1, energy2;

    energy_functions_top_level(image, prefitting_kernel, energy1, energy2);


    // LEVEL SET EVOLUTION

    namedWindow("debug_window", WINDOW_NORMAL);
    resizeWindow("debug_window", 500, 500);

    Mat_<double> LSF = initialLSF;

    for(int i=0; i<iterations_number; i++) {

      // work here

      LSF = active_contour_step(LSF, nu, timestep, mu, epsilon,
                                lambda1, lambda2,
                                energy1, energy2);


      // TODO - main loop end condition for changing area

      // print here - every X iteration

      int X = 10;

      if(i % X == 0) {
        Mat display_image = decorate_with_contours_from_acm_matrix(image, LSF);

        resize(display_image, display_image, Size(250,250));

        // text: iteration number X
        putText(display_image, "Iter: "+to_string(i), Point(20,20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(250));

        imshow("debug_window", display_image);

        int wait_time = 300;
        waitKey(wait_time);
      }


    }

    waitKey();

    return 0;
}
