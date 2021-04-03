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
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

using namespace std;
using namespace cv;
using namespace Eigen;
#define ld long double
// Overload

double NormTwo(Matrix<double, -1, -1> inputArray, int inputLength) { // Norm calculate
    long double norm = 0;
    for (int j = 0;j < inputLength;j++)
        norm += pow(abs(inputArray[j]), 2);
    return pow(norm, 0.5);
}

//friend Matrix<double, -1, -1> &operator= (Matrix<double, -1, -1> &a);
//Matrix<double, -1, -1> &operator = (Matrix<double, -1, -1> &a) {
//    Matrix<double ,-1, -1> b;
//    for (int i = 0;i < a.rows();i++)
//        for (int j = 0; j < a.cols();j++)
//            b(i,j) = a(i,j);
//    return b;
//}

Matrix<double , -1, -1> MirrorDescend(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    Matrix<double, 1, Dynamic> regret;
    Matrix<double, 1, Dynamic> regretBound;

    for (int INPUT_DIMENSION = INPUT_DIMENSION_LOWER; INPUT_DIMENSION < INPUT_DIMENSION_UPPER + 1 ;INPUT_DIMENSION++) {
        const int dimension = INPUT_DIMENSION;
        // INPUT GENERATION
        Matrix<double, 1, Dynamic> weightVector;
        weightVector.resize(1, INPUT_DIMENSION);
        for (int i = 0; i < INPUT_DIMENSION;i++)
            weightVector(0, i) = 1 / (INPUT_DIMENSION/1.0);
//        cout << weightVector << endl;
        Matrix<double, Dynamic, Dynamic> INPUT_MATRIX;
        INPUT_MATRIX.resize(INPUT_DIMENSION, INPUT_RANK);
        while(1) {
            for (int j = 0;j < INPUT_DIMENSION;j++)
                for (int i = 0;i < INPUT_RANK;i++)
                    INPUT_MATRIX(j, i) = (-1) + (double) ((rand() / (double) RAND_MAX) * 2);
            // Rank calculation
            FullPivLU<MatrixXd> Decomp(INPUT_MATRIX);
            if (Decomp.rank() == INPUT_RANK) break;
        }
//        cout << INPUT_MATRIX <<endl;
        double eta = sqrt(1 / ROUND);
        for (int T = 0 ; T < ROUND ; T++) {
            cout << T;
            // Generate the input
            Matrix<double, Dynamic, Dynamic> INPUT_V;
            INPUT_V.resize(1, INPUT_RANK);
            for (int i = 0; i < INPUT_RANK; i++)
                INPUT_V(0, i) = (-1) + (double) ((rand() / (double) RAND_MAX) * 2);
            Matrix<double, 1, Dynamic> Lt;

            Lt.resize(1, INPUT_DIMENSION);
            for (int i = 0; i < INPUT_DIMENSION; i++)
                for (int j = 0; j < INPUT_RANK; j++)
                    Lt(0, i) += INPUT_MATRIX(i, j) * INPUT_V(0, j);

            // Check Mark
            double biggestNorm = MAXFLOAT * (-1);
            for (int i = 0; i < INPUT_DIMENSION; i++)
                if (biggestNorm < Lt(0, i))
                    biggestNorm = Lt(0, i);
            for (int i = 0; i < INPUT_DIMENSION; i++)
                Lt(0, i) /= biggestNorm;

            Matrix<double, 1, Dynamic> LtNoise;
            LtNoise.resize(1, INPUT_DIMENSION);
            for (int j = 0; j < INPUT_DIMENSION; j++)
                LtNoise(0, j) = (double) (rand() / (double) RAND_MAX);
            for (int j = 0; j < INPUT_DIMENSION; j++) {
                LtNoise(0, j) /= (INPUT_DIMENSION * NormTwo(LtNoise, INPUT_DIMENSION));
                Lt(0, j) += LtNoise(0, j);
            }
            // Regret and Regret bound
            
            // Renew weight
            // OnlinePCA and MVEE


        }
    }
    return regretBound;
}

void MirrorDescendMain(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    Matrix<double, -1, -1> regret = MirrorDescend(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, INPUT_RANK, ROUND);
    double baseline[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
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