
#include "active_contour_method.h"

Mat curvature_central(const Mat & LSF);
Mat neumann_boundary_condition(const Mat &in);


// MAIN FUNCTION

Mat_<double> active_contour_step
(Mat_<double> LSF_init, double nu, double timestep, double mu, double epsilon, double lambda1, double lambda2, Mat_<double> energy1, Mat_<double> energy2) {

    // TODO annotate this signature


    Mat_<double> LSF = LSF_init.clone();

    LSF = neumann_boundary_condition(LSF);

    Mat_<double> K = curvature_central(LSF);

    Mat_<double> DrcU = 1 / (LSF.mul(LSF) + epsilon * epsilon) * epsilon / M_PI;


    // H=0.5*(1+(2/pi)*atan(u./epsilon));

    Mat H(LSF.size(), CV_64FC1, Scalar::all(0));

    for(int i=0; i<H.rows; i++){
      for(int j=0; j<H.cols; j++){
        // prepare
	    double LSF_elem = LSF.at<double>(Point(j,i));

        // -- calculate --
	    double H_elem = 0.5 * (1 + (2/M_PI) * atan(LSF_elem / epsilon));

        // write
	    H.at<double>(Point(j, i)) = H_elem;
      }
    }


    // LPFterm=-DrcU.*(e1.*lambda1-e2.*lambda2);

    Mat_<double> LPFterm(DrcU.size(), CV_64FC1);

    for(int i=0; i<LPFterm.rows; i++) {
      for(int j=0; j<LPFterm.cols; j++) {
        // prepare
        double DrcU_elem = DrcU.at<double>(Point(j,i));
        double e1_elem = energy1.at<double>(Point(j,i));
        double e2_elem = energy2.at<double>(Point(j,i));

        // -- calculate --
        double new_elem = -DrcU_elem * (e1_elem * lambda1 - e2_elem * lambda2);

        // write
        LPFterm.at<double>(Point(j,i)) = new_elem;
      }
    }


    // PenaltyTerm=mu*(4*del2(u)-K);

    Mat_<double> PenaltyTerm;
    Mat_<double> u_laplacian;

    // NOT SURE IF THIS IS SAME AS del2(u) !!!
    Laplacian(LSF, u_laplacian, -1, 1, 0.25);

//    cout << "next_laplacian:" << endl;
//    cout << u_laplacian(Range(30,40), Range(30,40));
//    cout << endl;

    PenaltyTerm = mu * (4 * u_laplacian - K);


    // LengthTerm=nu.*DrcU.*K;

    Mat_<double> LengthTerm = DrcU.mul(K) * nu;

    LSF = LSF + timestep * (LengthTerm + PenaltyTerm + LPFterm);

    return LSF;
}

// HELPER

Mat curvature_central(const Mat & LSF) {

    // return K; same size and type as LSF
    Mat K(LSF.size(), LSF.type());

    Mat ux, uy;
    Sobel(LSF, ux, -1, 1, 0, 1, 0.5);
    Sobel(LSF, uy, -1, 0, 1, 1, 0.5);

    Mat normDu = (ux.mul(ux) + uy.mul(uy)) + 1e-10;
    sqrt(normDu, normDu);

    Mat Nx = ux.mul(1 / normDu);
    Mat Ny = uy.mul(1 / normDu);

    Mat nxx, nyy;

    Sobel(Nx, nxx, -1, 1, 0, 1, 0.5);
    Sobel(Ny, nyy, -1, 0, 1, 1, 0.5);

    K = nxx + nyy;

    return K;
}

// HELPER

Mat neumann_boundary_condition(const Mat &in) {

    // copies values that are 2 pixels away from borders onto borders

    Mat out = in.clone();

    // 4 corners

    out.at<float>(Point(0, 0)) = out.at<float>(Point(2, 2));
    out.at<float>(Point(out.cols - 1, 0)) = out.at<float>(Point(out.cols - 3, 2));
    out.at<float>(Point(0, out.rows - 1)) = out.at<float>(Point(2, out.rows - 3));
    out.at<float>(Point(out.cols - 1, out.rows - 1)) =
            out.at<float>(Point(out.cols - 3, out.rows - 3));

    // top/bottom edges (without corners)

    Rect top_edge(Point(1, 0), Size(out.cols - 2, 1));
    Rect bottom_edge(Point(1, out.rows - 1), Size(out.cols - 2, 1));
    Rect one_of_top_rows(Point(1, 2), Size(out.cols - 2, 1));
    Rect one_of_bottom_rows(Point(1, out.rows - 3), Size(out.cols - 2, 1));

    out(one_of_top_rows).copyTo(out(top_edge));
    out(one_of_bottom_rows).copyTo(out(bottom_edge));

    // left/right edges (without corners)

    Rect left_edge(Point(0, 1), Size(1, out.rows - 2));
    Rect right_edge(Point(out.cols - 1, 1), Size(1, out.rows - 2));
    Rect one_of_left_cols(Point(2, 1), Size(1, out.rows - 2));
    Rect one_of_right_cols(Point(out.cols - 3, 1), Size(1, out.rows - 2));

    out(one_of_left_cols).copyTo(out(left_edge));
    out(one_of_right_cols).copyTo(out(right_edge));

    return out;
}

// ADDITIONAL UTILITY

Mat_<uchar> decorate_with_contours_from_acm_matrix
(const Mat_<uchar> &image, const Mat_<double> &LSF) {
    // draw contour based on LSF
    // (contours are drawn around areas with negative values)

    // it tries to imitate Matlab behaviour for drawing contours

    Mat new_image = image.clone();

    Mat contour_mat;

    // only take negative
    double limit = 0.0;
    threshold(LSF, contour_mat, limit, 1, THRESH_BINARY);

    // convert to uchars
    contour_mat.convertTo(contour_mat, CV_8UC1, 255.0);

    // swap negative/positive
    contour_mat = contour_mat * -1 + 255;

    // find contours (positive areas)
    vector<vector<Point>> contours;
    findContours(contour_mat, contours, RETR_LIST, CHAIN_APPROX_NONE);

    Mat_<uchar> clean_img = Mat::zeros(image.size(), CV_8UC1);

    int which_contour = -1;     // means 'all'
    Scalar colour(250);
    drawContours(new_image, contours, which_contour, colour);

    return new_image;
}
