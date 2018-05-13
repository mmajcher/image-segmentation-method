
#include "local_prefitting_functions.h"

void local_prefitting_functions(const Mat_<uchar> & image, const Mat & kernel, Mat &f1, Mat &f2) {

    // KK - helper kernel?
    Mat KK = kernel.clone();
    KK = KK * kernel.rows * kernel.cols;

    // r - border size (for calculating kernel values at edges of image)
    int r = (kernel.rows - 1) / 2;


    // prepare extended image (image -> image_n)

    // image_n is original image extended with boundaries
    // it gets additional boundaries of size r
    CV_Assert(image.type() == CV_8UC1);
    Mat image_n = Mat::zeros(image.rows + 2 * r, image.cols + 2 * r, CV_8UC1);

    // fill original (central) part of image_n

    Rect original_part(Point(r, r), image.size());
    image.copyTo(image_n(original_part));

    // fill all boundaries (their width is r) of image_n

    Mat top_border = image_n(Rect(Point(r, 0), Size(image.cols, r)));
    Mat original_top = image(Rect(Point(0, 0), Size(image.cols, r)));
    original_top.copyTo(top_border);

    Mat right_border = image_n(Rect(Point(r + image.cols, 0), Size(r, image.rows + r)));
    Mat right_fill = image_n(Rect(Point(image.cols, 0), Size(r, image.rows + r)));
    right_fill.copyTo(right_border);

    Mat bottom = image_n(Rect(Point(r, image.rows + r), Size(image.cols + r, r)));
    Mat bottom_fill = image_n(Rect(Point(r, image.rows), Size(image.cols + r, r)));
    bottom_fill.copyTo(bottom);

    Mat left = image_n(Rect(Point(0, 0), Size(r, image.rows + 2 * r)));
    Mat left_fill = image_n(Rect(Point(r, 0), Size(r, image.rows + 2 * r)));
    left_fill.copyTo(left);


    // initialize outputs & helpers

    f1 = Mat::zeros(image_n.size(), CV_32FC1);
    f2 = Mat::zeros(image_n.size(), CV_32FC1);
    Mat s1 = Mat::zeros(image_n.size(), CV_32FC1);
    Mat s2 = Mat::zeros(image_n.size(), CV_32FC1);
    Mat image_mean_values = Mat::zeros(image_n.size(), CV_32FC1);


    // the loop; prepare prefitting matrix

    for (int i = r; i < image.rows + r; i++) {
        for (int j = r; j < image.cols + r; j++) {
            Rect current_rect(Point(j - r, i - r), Size(2 * r + 1, 2 * r + 1));
            Mat current_part = image_n(current_rect).clone();
            current_part.convertTo(current_part, CV_32FC1);


            // mean value of non-zero elements

            float sum_non_zero = 0;
            int number_of_non_zero = 0;

            for (int i = 0; i < current_part.rows; i++) {
                for (int j = 0; j < current_part.cols; j++) {
                    // sum non-zero elements
                    float elem = current_part.at<float>(Point(j, i));
                    if (elem != 0) {
                        sum_non_zero += elem;
                        number_of_non_zero++;
                    }
                }
            }

            float mean_value = sum_non_zero / number_of_non_zero;

            image_mean_values.at<float>(Point(j, i)) = mean_value;


            // partition current_part into values greater and lesser than mean_value

            Mat only_lower_than_mean = Mat::zeros(current_part.size(), CV_32FC1);
            current_part.copyTo(only_lower_than_mean, ((current_part > 0) & (current_part <= mean_value)));

            Mat only_greater_than_mean = Mat::zeros(current_part.size(), CV_32FC1);
            current_part.copyTo(only_greater_than_mean, current_part >= mean_value);


            // partition KK to mirror above partitioning

            Mat KK_only_lower_than_mean = Mat::zeros(current_part.size(), CV_32FC1);
            KK.copyTo(KK_only_lower_than_mean, ((current_part > 0) & (current_part <= mean_value)));

            Mat KK_only_greater_than_mean = Mat::zeros(current_part.size(), CV_32FC1);
            KK.copyTo(KK_only_greater_than_mean, current_part >= mean_value);


            // point-wise multiply

            multiply(only_greater_than_mean, KK_only_greater_than_mean,
                    only_greater_than_mean);

            multiply(only_lower_than_mean, KK_only_lower_than_mean,
                    only_lower_than_mean);


            // calculate f1 f2

            float f1_current = sum(only_lower_than_mean)[0] /
                    (sum(KK_only_lower_than_mean)[0] + numeric_limits<float>::epsilon());

            float f2_current = sum(only_greater_than_mean)[0] /
                    (sum(KK_only_greater_than_mean)[0] + numeric_limits<float>::epsilon());

            f1.at<float>(Point(j, i)) = f1_current;
            f2.at<float>(Point(j, i)) = f2_current;


            // calculate s1 s2

            Mat temp1 = Mat::zeros(current_part.size(), CV_32FC1);
            temp1.setTo(f1_current, only_lower_than_mean != 0);

            temp1 = temp1 - only_lower_than_mean;

            multiply(temp1, temp1, temp1);

            float s1_current = sum(temp1)[0];


            Mat temp2 = Mat::zeros(current_part.size(), CV_32FC1);
            temp2.setTo(f2_current, only_greater_than_mean != 0);

            temp2 = temp2 - only_greater_than_mean;

            multiply(temp2, temp2, temp2);

            float s2_current = sum(temp2)[0];

            s1.at<float>(Point(j, i)) = s1_current;
            s2.at<float>(Point(j, i)) = s2_current;

        }
    }

    f1 = f1(original_part);
    f2 = f2(original_part);
    s1 = s1(original_part);
    s2 = s2(original_part);

}

void energy_functions_from_prefiting_functions(const Mat_<uchar> & image, const Mat & prefitting_kernel, const Mat & prefit1, const Mat & prefit2, Mat & energy1, Mat & energy2) {

    Mat prefitting_1 = prefit1.clone();
    Mat prefitting_2 = prefit2.clone();

    Mat image_float = image.clone();
    image_float.convertTo(image_float, CV_32FC1);


	// e1=Img.*Img.*imfilter(ones(size(Img)),K,'replicate')-2.*Img.*imfilter(f1,K,'replicate')+imfilter(f1.^2,K,'replicate');
	// e2=Img.*Img.*imfilter(ones(size(Img)),K,'replicate')-2.*Img.*imfilter(f2,K,'replicate')+imfilter(f2.^2,K,'replicate');


    // ENERGY FUNCTION 1

    Mat imfilter1;
    filter2D(Mat::ones(image.size(), CV_32FC1), imfilter1, -1, prefitting_kernel);

    Mat imfilter2;
    prefitting_1.convertTo(prefitting_1, CV_8UC1);
    filter2D(prefitting_1, imfilter2, CV_32FC1, prefitting_kernel,
            Point(-1, -1), 0, BORDER_REPLICATE);

    Mat imfilter3;
    prefitting_1.convertTo(prefitting_1, CV_16UC1);
    filter2D(prefitting_1.mul(prefitting_1), imfilter3, CV_32FC1, prefitting_kernel,
            Point(-1, -1), 0, BORDER_REPLICATE);

    energy1 = Mat(image.size(), CV_32FC1);

    for(int i=0; i < image_float.rows; i++) {
    	for(int j=0; j < image_float.cols; j++) {
    		Point current_point(j, i);

    		float part_one = image_float.at<float>(current_point)
    				* image_float.at<float>(current_point) * imfilter1.at<float>(current_point);

    		float part_two = -2 * image_float.at<float>(current_point) * imfilter2.at<float>(current_point);

    		float part_three = imfilter3.at<float>(current_point);

    		energy1.at<float>(current_point) = part_one + part_two + part_three;
    	}
    }

    // TODO rewrite energy function 2


    // ENERGY FUNCTION 2

    Mat energy2_expr1;

    Mat energy2_filter1;
    filter2D(Mat::ones(image.size(), CV_32FC1), energy2_filter1, -1, prefitting_kernel);

    energy2_expr1 = image_float.clone();
    energy2_expr1 = (energy2_expr1.mul(energy2_expr1));
    energy2_expr1 = energy2_expr1.mul(energy2_filter1);


    Mat energy2_expr2;

    Mat energy2_filter2;
    prefitting_2.convertTo(prefitting_2, CV_8UC1);
    filter2D(prefitting_2, energy2_filter2, CV_32FC1, prefitting_kernel,
            Point(-1, -1), 0, BORDER_REPLICATE);

    energy2_expr2 = image_float.clone();
    energy2_expr2 = energy2_expr2 * 2;
    energy2_expr2 = energy2_expr2.mul(energy2_filter2);


    Mat energy2_expr3;

    prefitting_2.convertTo(prefitting_2, CV_16UC1);
    filter2D(prefitting_2.mul(prefitting_2), energy2_expr3, CV_32FC1, prefitting_kernel,
            Point(-1, -1), 0, BORDER_REPLICATE);


    energy2 = energy2_expr1 - energy2_expr2 + energy2_expr3;
}
