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
using namespace std;
using namespace cv;

#define INPUT_DIMENSION_UPPER 100
#define INPUT_DIMENSION_LOWER 4
#define INPUT_RANK 3
#define ROUND 100

double *MirrorDescend() {
    long double regret[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    
}

void MirrorDescendMain() {
    long double *regret = MirrorDescend();
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