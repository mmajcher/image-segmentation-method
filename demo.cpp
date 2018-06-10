#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <vector>

#include "active_contour_method.h"
#include "local_prefitting_functions.h"

using namespace cv;
using namespace std;

#define FRAME_PRESENTATION_TIME 100


Mat _decorate_with_contours(const Mat &image, vector<vector<Point>> contours) {

    Mat decorated = image.clone();

    int which_contour = -1;     // means 'all'
    Scalar colour(250);
    drawContours(decorated, contours, which_contour, colour);

    return decorated;
}

Mat _annotate(String text, const Mat &image) {
    Mat annotated = image.clone();

    putText(annotated, text, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5,
            Scalar(250));

    return annotated;
}

Mat _read_initial_lsf_from_file(Size image_size, String filename) {

    // TODO read lsf from file

    return Mat::zeros(image_size, CV_64FC1);
}

void _write_contours_to_file(vector<vector<Point>> contours, String filename) {

    // TODO write contours to file
}



void _image_display(const Mat &image, String display, String caption) {

    Mat display_image = image;

    if(caption.length() > 0) {
        display_image = image.clone();

        putText(display_image, caption, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5,
                Scalar(250));
    }

    imshow(display, display_image);
    waitKey(FRAME_PRESENTATION_TIME);
}

bool _should_display_image(int iteration) {
    if(iteration < 5 )
        return true;

    if(iteration % 10 == 0)
        return true;

    return false;
}



int main() {

    // TODO CommandLineParser

    short OPTION_READ_INITIAL_LSF = 0;
    short OPTION_SAVE_ALL = 0;
    short OPTION_SAVE_CONTOURS = 0;
    short OPTION_SAVE_LAST = 0;
    String OPTION_SAVE_CONTOURS_FILENAME = "saved_contours";


    namedWindow("display", WINDOW_NORMAL);
    resizeWindow("display", 500, 500);


    // READ AND DISPLAY IMAGE

    Mat_<uchar> image = imread("images/2.bmp", IMREAD_GRAYSCALE);

    _image_display(image, "display", "image");


    // INITIAL LSF
    // -- mat<double> of same size as image
    // -- contours are drawn around areas with negative values

    Mat_<double> initialLSF;

    if(OPTION_READ_INITIAL_LSF) {
        // TODO read Initial lsf from file
    }
    else {
        // default; random rectangle
        initialLSF = Mat::ones(image.size(), CV_64FC1);
        Rect some_rectangle(Point(40, 15), Size(20, 20));
        initialLSF(some_rectangle).setTo(-1.0);
    }

    // -- or only ones? (or zeros?)
    // initialLSF = Mat::ones(image.size(), CV_64FC1);

    vector<vector<Point>> initial_contours = acm_get_contours(initialLSF);

    _image_display(_decorate_with_contours(image, initial_contours), "display",
            "initial contour");


    // ACM PARAMETERS

    int mu = 1;
    double nu = 0.01 * 255 * 255;
    int lambda1 = 1;
    int lambda2 = lambda1;
    double epsilon = 1.0;
    double timestep = 0.02;
    int iterations_number = 200;
    double sigma = 2; // control local size


    // ENERGY FUNCTIONS based on local prefitting functions

    int gauss_kernel_size = round(2 * sigma) * 2 + 1;
    Mat_<double> gauss_kernel_1d = getGaussianKernel(gauss_kernel_size, sigma, CV_64FC1);
    Mat_<double> gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t();

    Mat_<double> prefitting_kernel = gauss_kernel_2d;
    Mat_<double> energy1, energy2;

    energy_functions_top_level(image, prefitting_kernel, energy1, energy2);


    // ACM LOOP

    Mat_<double> LSF = initialLSF;

    for(int i=1; i<=iterations_number; i++) {

      // contours evolution

      LSF = acm_advance(LSF, nu, timestep, mu, epsilon,
                                lambda1, lambda2,
                                energy1, energy2);


      // print here - first 5 and then every Xth iteration

      if(_should_display_image(i) || OPTION_SAVE_ALL ) {

        // contours
        vector<vector<Point>> contours = acm_get_contours(LSF);
        Mat display_image = _decorate_with_contours(image, contours);

        if(OPTION_SAVE_ALL) {
            // write_image_to_file
        }

        if(_should_display_image(i)) {
            // annotation
            String annotation = "iter: " + to_string(i);
            display_image = _annotate(annotation, display_image);

            // display
            resize(display_image, display_image, Size(250,250));
            imshow("display", display_image);
            waitKey(FRAME_PRESENTATION_TIME);
        }
      }
    }


    if(OPTION_SAVE_CONTOURS) {
    // TODO save contours to file if OPTION_SAVE_CONTOURS
        _write_contours_to_file(acm_get_contours(LSF), OPTION_SAVE_CONTOURS_FILENAME);
    }

    if(OPTION_SAVE_LAST) {
    // TODO save image to file if OPTION_SAVE_LAST
    //       _write_image_fo_file()
    }

    waitKey();

    return 0;
}
