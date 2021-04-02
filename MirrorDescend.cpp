//
// Created by holmes on 2021/3/29.
//

# include <iostream>
# include <math.h>
# include "MVEE.h"
# include "OnlinePCA.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MirrorDescend.h"
#include <Eigen/Dense>"
#include <Eigen/Core>"
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace cv;
using namespace Eigen;


#define ld long double

Matrix<ld , -1, -1> MirrorDescend(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    Matrix<ld, Dynamic, 1> regret;
    Matrix<ld, Dynamic, 1> regretBound;
    // INPUT GENERATION
    

}

void MirrorDescendMain(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    Matrix<long double, -1, -1> regret = MirrorDescend(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, INPUT_RANK, ROUND);
    long double baseline[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    int TIME_ROUND[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];

    for (int i = 0; i < (INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER); i++) {
        baseline[i] = i + 3;
        TIME_ROUND[i] = i + 1;
    }
    Mat img = Mat::zeros(Size(800, 600), CV_8UC3);
    img.setTo(255);
    Point point[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    for (int i = 0; i < (INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER); i++) {
        Point p(TIME_ROUND[i], regret[i]); point[i] = p;}
    for (int i = 0; i < (INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER) - 1; i++)
        line(img, point[i], point[i+1], Scalar(0, 0, 255), 2);
    imshow("regert - N", img);

    waitKey();
    cout << "Done" << endl;
}