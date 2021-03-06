
#include "local_prefitting_functions.h"

void local_prefitting_functions(const Mat_<uchar> & image, const Mat_<double> & kernel, Mat_<double> &f1, Mat_<double> &f2);

void energy_functions_from_prefiting_functions
(const Mat_<uchar> & image, const Mat_<double> & prefitting_kernel,
 const Mat_<double> & prefit1, const Mat_<double> & prefit2,
 Mat & energy1, Mat & energy2);



void energy_functions_top_level(const Mat_<uchar> &image, const Mat_<double> &kernel, Mat_<double> &energy_1, Mat_<double> &energy_2) {


    // step one - LOCAL PREFITTING FUNCTIONS

    Mat_<double> prefitting_1;
    Mat_<double> prefitting_2;

    local_prefitting_functions
        (image, kernel, prefitting_1, prefitting_2);


    // step two - ENERGY FUNCTIONS based on local prefitting functions

    energy_functions_from_prefiting_functions
        (image, kernel, prefitting_1, prefitting_2, energy_1, energy_2);

}


void local_prefitting_functions(const Mat_<uchar> & image, const Mat_<double> & kernel, Mat_<double> &f1, Mat_<double> &f2) {

    // KK - additional helper kernel
    Mat_<double> KK = kernel * kernel.rows * kernel.cols;

    // r - border size (for calculating kernel values at edges of image)
    int r = (kernel.rows - 1) / 2;


    // PREPARE EXTENDED IMAGE (image -> image_extended)

    /*
      image_extended is original image extended with boundaries;
      it gets additional boundaries of size r
    */

    Mat image_extended = Mat::zeros(image.rows + 2 * r, image.cols + 2 * r, CV_8UC1);

    int border = r;
    copyMakeBorder(image, image_extended, border, border,
                   border, border, BORDER_REPLICATE);

    Rect original_part_of_image_extended(Point(r, r), image.size());


    // INITIALIZE OUTPUTS & HELPERS

    f1 = Mat::zeros(image_extended.size(), CV_64FC1);
    f2 = Mat::zeros(image_extended.size(), CV_64FC1);
    Mat s1 = Mat::zeros(image_extended.size(), CV_64FC1);
    Mat s2 = Mat::zeros(image_extended.size(), CV_64FC1);

    Mat image_mean_values = Mat::zeros(image_extended.size(), CV_64FC1);

    Size window_size(2 * r + 1, 2 * r + 1);


    // THE LOOP; PREPARE PREFITTING MATRIX

    // -- loop over extended image
    for (int y = r; y < image.rows + r; y++) {
        for (int x = r; x < image.cols + r; x++) {

            Rect current(Point(x - r, y - r), window_size);
            Mat_<double> window = image_extended(current);

            // mean value in the window (only non-zero elements)
            double mean_value = mean(window, window != 0)[0];

            Mat_<double> weighted_window = window.mul(KK);

            double sum_weighted_elems_below_mean = 0;
            double sum_weighted_elems_above_mean = 0;
            double sum_KK_below_mean = 0;
            double sum_KK_above_mean = 0;

            // -- loop over current window
            for(int y = 0; y < window.rows; y++) {
                for(int x = 0; x < window.cols; x++) {
                    Point point(x, y);

                    double elem = window.at<double>(point);
                    double KK_elem = KK.at<double>(point);
                    double weighted_elem = weighted_window.at<double>(point);

                    if(elem >= mean_value) {
                        sum_weighted_elems_above_mean += weighted_elem;
                        sum_KK_above_mean += KK_elem;
                    }

                    if (elem <= mean_value && elem > 0) {
                        sum_weighted_elems_below_mean += weighted_elem;
                        sum_KK_below_mean += KK_elem;
                    }
                }
            }

            // CALCULATE f1 f2

            double f1_elem = sum_weighted_elems_below_mean /
                (sum_KK_below_mean + numeric_limits<double>::epsilon());

            double f2_elem = sum_weighted_elems_above_mean /
                (sum_KK_above_mean + numeric_limits<double>::epsilon());

            f1.at<double>(Point(x, y)) = f1_elem;
            f2.at<double>(Point(x, y)) = f2_elem;


        }
    }

    f1 = f1(original_part_of_image_extended);
    f2 = f2(original_part_of_image_extended);

}

void energy_functions_from_prefiting_functions
(const Mat_<uchar> & image, const Mat_<double> & prefitting_kernel,
 const Mat_<double> & prefit1, const Mat_<double> & prefit2,
 Mat & energy1, Mat & energy2) {


    Mat prefitting_1 = prefit1.clone();
    Mat prefitting_2 = prefit2.clone();


    // ENERGY FUNCTION 1

	// e1=Img.*Img.*imfilter(ones(size(Img)),K,'replicate')-2.*Img.*imfilter(f1,K,'replicate')+imfilter(f1.^2,K,'replicate');

    Mat energy1_imfilter1;
    filter2D(Mat::ones(image.size(), CV_64FC1), energy1_imfilter1,
             CV_64FC1, prefitting_kernel);

    Mat energy1_imfilter2;
    filter2D(prefitting_1, energy1_imfilter2,
             CV_64FC1, prefitting_kernel,
             Point(-1, -1), 0, BORDER_REPLICATE);

    Mat energy1_imfilter3;
    filter2D(prefitting_1.mul(prefitting_1), energy1_imfilter3,
             CV_64FC1, prefitting_kernel,
             Point(-1, -1), 0, BORDER_REPLICATE);


    energy1 = Mat(image.size(), CV_64FC1);

    for(int y=0; y < image.rows; y++) {
    	for(int x=0; x < image.cols; x++) {
    		Point current_point(x, y);

    		double part_one = (double) image.at<uchar>(current_point)
                * image.at<uchar>(current_point) * energy1_imfilter1.at<double>(current_point);

    		double part_two = -2.0 * image.at<uchar>(current_point)
                * energy1_imfilter2.at<double>(current_point);

    		double part_three = energy1_imfilter3.at<double>(current_point);

    		energy1.at<double>(current_point) = part_one + part_two + part_three;
    	}
    }



    // ENERGY FUNCTION 2

	// e2=Img.*Img.*imfilter(ones(size(Img)),K,'replicate')-2.*Img.*imfilter(f2,K,'replicate')+imfilter(f2.^2,K,'replicate');

    Mat energy2_imfilter1;
    filter2D(Mat::ones(image.size(), CV_64FC1), energy2_imfilter1,
             CV_64FC1, prefitting_kernel);

    Mat energy2_imfilter2;
    filter2D(prefitting_2, energy2_imfilter2,
             CV_64FC1, prefitting_kernel,
             Point(-1, -1), 0, BORDER_REPLICATE);

    Mat energy2_imfilter3;
    filter2D(prefitting_2.mul(prefitting_2), energy2_imfilter3,
             CV_64FC1, prefitting_kernel,
             Point(-1, -1), 0, BORDER_REPLICATE);


    energy2 = Mat(image.size(), CV_64FC1);

    for(int y=0; y < image.rows; y++) {
    	for(int x=0; x < image.cols; x++) {
    		Point current_point(x, y);

    		double part_one = (double) image.at<uchar>(current_point)
                * image.at<uchar>(current_point) * energy2_imfilter1.at<double>(current_point);

    		double part_two = -2.0 * image.at<uchar>(current_point)
                * energy2_imfilter2.at<double>(current_point);

    		double part_three = energy2_imfilter3.at<double>(current_point);

    		energy2.at<double>(current_point) = part_one + part_two + part_three;
    	}
    }

}
