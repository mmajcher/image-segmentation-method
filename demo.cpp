#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <vector>

#include "active_contour_method.h"
#include "local_prefitting_functions.h"

using namespace cv;
using namespace std;

#define FRAME_PRESENTATION_TIME -1


Mat _get_image_decorated_with_contours(const Mat &image, vector<vector<Point>> contours);
void _display_image(const Mat &image, string display, string caption);

Mat _read_initial_lsfmatrix_from_file(Size image_size, string filename);
void _write_contours_to_file(vector<vector<Point>> contours, string filename);


bool _should_display_image(int iteration) {
    if(iteration < 5 )
        return true;

    if(iteration % 10 == 0)
        return true;

    return false;
}



int main(int argc, char** argv) {

    string cmdline_options =
            "{help h | | print this message }"
            "{initial-contour | <none> | specify file with initial contour description }"
            "{save-all-images-to-dir | <none> | specify directory to save images from every iteration into }"
            "{save-final-contours | <none> | specify file to save final contour points }"
            "{save-last-image | <none> | specify file for saving last image}"

            "{mu | 1.0 | acm param}"
            "{nu | 650.0 | acm param}"
            "{lambda1 | 1.0 | acm param}"
            "{lambda2 | 1.0 | acm param}"
            "{epsilon | 1.0 | acm param}"
            "{timestep | 0.02 | acm param}"
            "{iterations | 200 | acm param}"
            "{sigma | 2.0 | acm param}"
            ;

    CommandLineParser parser(argc, argv, cmdline_options);

    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    namedWindow("display", WINDOW_NORMAL);

    // READ AND DISPLAY IMAGE

    Mat_<uchar> image = imread("images/2.bmp", IMREAD_GRAYSCALE);

    _display_image(image, "display", "image");


    // ==== INITIAL CONTOUR

    // -- mat<double> of same size as image
    // -- contours are drawn around areas with negative values

    Mat_<double> initialLSF;

    if(parser.has("initial-contour")) {
        string initial_contour_filename = parser.get<String>("initial-contour");
        cout << "initial contour filename: " << initial_contour_filename << endl;
        initialLSF = _read_initial_lsfmatrix_from_file(image.size(), initial_contour_filename);
    }
    else {
        // default; random rectangle
        initialLSF = Mat::ones(image.size(), CV_64FC1);
        Rect some_rectangle(Point(40, 15), Size(20, 20));
        initialLSF(some_rectangle).setTo(-1.0);
    }

    vector<vector<Point>> initial_contours = acm_get_contours(initialLSF);

    _display_image(_get_image_decorated_with_contours(image, initial_contours), "display",
            "initial contour");


    // ==== PARAMS

    double mu = parser.get<double>("mu");
    double nu = parser.get<double>("nu");
    double lambda1 = parser.get<double>("lambda1");
    double lambda2 = parser.get<double>("lambda2");
    double epsilon = parser.get<double>("epsilon");
    double timestep = parser.get<double>("timestep");
    int iterations_number = parser.get<double>("iterations");
    double sigma = parser.get<double>("sigma");


    // ==== LOCAL PREFITTING ENERGY FUNCTIONS

    int gauss_kernel_size = round(2 * sigma) * 2 + 1;
    Mat_<double> gauss_kernel_1d = getGaussianKernel(gauss_kernel_size, sigma, CV_64FC1);
    Mat_<double> gauss_kernel_2d = gauss_kernel_1d * gauss_kernel_1d.t();

    Mat_<double> prefitting_kernel = gauss_kernel_2d;
    Mat_<double> energy1, energy2;

    energy_functions_top_level(image, prefitting_kernel, energy1, energy2);


    // ==== ACM LOOP

    Mat_<double> LSF = initialLSF;

    for (int i = 1; i <= iterations_number; i++) {

        LSF = acm_advance(LSF, nu, timestep, mu, epsilon, lambda1, lambda2,
                energy1, energy2);

        if(_should_display_image(i) || parser.has("save-all-images-to-dir")) {

            // 1 - get contours

            vector<vector<Point>> contours = acm_get_contours(LSF);
            Mat display_image = _get_image_decorated_with_contours(image,
                    contours);

            // 2 - save frame

            if(parser.has("save-all-images-to-dir")) {
                string save_all_dir = parser.get<String>( "save-all-images-to-dir");
                char iter_num[5];
                sprintf(iter_num, "%04d", i);
                string filename = save_all_dir + "/" + "iter_" + iter_num + ".jpg";
                cout << "saving image: " << filename << endl;
                imwrite(filename, display_image);
            }

            // 3 - display frame

            if(_should_display_image(i)) {
                // annotation
                string annotation = "iter: " + to_string(i);

                // display
                _display_image(display_image, "display", annotation);
            }
        }

    }


    // ==== FINALIZE

    if(parser.has("save-final-contours")) {
        string final_contours_filename = parser.get<String>("save-final-contours");
        _write_contours_to_file(acm_get_contours(LSF), final_contours_filename);
    }

    if(parser.has("save-last-image")) {
        string final_image_filename = parser.get<String>("save-last-image");

        vector<vector<Point>> final_contours = acm_get_contours(LSF);

        imwrite(final_image_filename, _get_image_decorated_with_contours(image, final_contours));
    }

    waitKey();

    return 0;
}


Mat _get_image_decorated_with_contours(const Mat &image, vector<vector<Point>> contours) {

    Mat decorated = image.clone();

    int which_contour = -1;     // means 'all'
    Scalar colour(250);
    drawContours(decorated, contours, which_contour, colour);

    return decorated;
}


void _display_image(const Mat &image, string display, string caption) {

    Mat display_image = image;

    if(caption.length() > 0) {
        display_image = image.clone();

        putText(display_image, caption, Point(10,10), FONT_HERSHEY_SIMPLEX, 0.3,
                Scalar(250));
    }

    imshow(display, display_image);
    waitKey(FRAME_PRESENTATION_TIME);
}

Mat _read_initial_lsfmatrix_from_file(Size image_size, string filename) {

    ifstream in(filename, ifstream::in);

    string shape;
    in >> shape;

    cout << "initial contours, shape: " << shape << endl;

    Mat_<double> initial_lsf = Mat::ones(image_size, CV_64FC1);

    if(shape == "rectangle") {
        Point top_left, bot_right;

        in >> top_left.x >> top_left.y >> bot_right.x >> bot_right.y;

        cout << "rectangle " << top_left << bot_right << endl;
        initial_lsf(Rect(top_left, bot_right)).setTo(-1.0);
    }
    else if(shape == "ellipse") {
        // TODO ellipse
    }
    else if(shape == "circle") {
        // TODO circle
    }
    else if(shape == "contour") {
        // TODO contour points
    }
    else {
        cout << "unknown initial-contour shape; please use: rectangle ellipse circle" << endl;
        exit(1);
    }

    return initial_lsf;
}

void _write_contours_to_file(vector<vector<Point>> contours, string filename) {

    // TODO write contours to file
}
