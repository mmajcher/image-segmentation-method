
#include "draw_contour_from_acm_matrix.h"


Mat decorate_with_contours_from_acm_matrix(const Mat& image, const Mat& LSF) {
    // draw contour based on LSF
    // (contours are drawn around areas with negative values)
    
    // it tries to imitate Matlab behaviour
    
    CV_Assert(LSF.type() == CV_32FC1);
    CV_Assert(image.type() == CV_8UC1);
    

    Mat new_image = image.clone();

    Mat contour_mat;
   
    // only take negative
    threshold(LSF, contour_mat, 0.0, 1, THRESH_BINARY);
    contour_mat.convertTo(contour_mat, CV_8U, 255.0);

    // swap negative/positive
    contour_mat = contour_mat * -1 + 255;

    // find contours (positive areas)
    vector<vector < Point>> contours;
    findContours(contour_mat, contours, RETR_LIST, CHAIN_APPROX_NONE);

    Mat clean_img = Mat::zeros(image.size(), CV_8U);

    drawContours(new_image, contours, -1, Scalar(250));

    return new_image;
}
