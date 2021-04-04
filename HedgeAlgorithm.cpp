//
// Created by holmes on 2021/3/29.
//
// N changing version, but it's easy to get the Round one
#include <iostream>
#include "math.h"
#include "time.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "HedgeAlgorithm.h"

using namespace std;
using namespace cv;

double NormTwo(double*inputArray, int inputLength) { // Norm calculate
    double norm = 0;
    for (int j = 0;j < inputLength;j++)
        norm += pow(abs(inputArray[j]), 2);
    return pow(norm, 0.5);
}

void HedgeAlgorithm(int INPUT_DIMENSION_LOWER, int  INPUT_DIMENSION_UPPER, int ROUND) {
    double regret[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    regret[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER] = {0};
    int regretI = 0;
//    srand(time(NULL));
    for (int INPUT_DIMENSION = INPUT_DIMENSION_LOWER ; INPUT_DIMENSION <= INPUT_DIMENSION_UPPER ; INPUT_DIMENSION++) {
        cout << "DIMENSION: " << INPUT_DIMENSION <<endl;
        // Init weight vector
        double weightVector[INPUT_DIMENSION];
        for (int i = 0; i < INPUT_DIMENSION;i++) {
            weightVector[i] = 1 / (INPUT_DIMENSION / 1.0);
        }
        double currentRegret = 0.0;
        double inputTSum[INPUT_DIMENSION];
        inputTSum[INPUT_DIMENSION] = {0};
        double minuend = 0.0;

        for (int T = ROUND; T < ROUND + 1;T++)
            for (int t = 0; t < T ; t++) {
                double eta = pow(log(INPUT_DIMENSION) / (T / 1.0), 0.5);
                double inputVector[INPUT_DIMENSION];
                inputVector[INPUT_DIMENSION] = {0};
                for (int i = 0;i < INPUT_DIMENSION; i++) {
                    inputVector[i] = (-1) + (double) ((rand() / (double) RAND_MAX) * 2);
                }

                double LtNoise[INPUT_DIMENSION];
                LtNoise[INPUT_DIMENSION] = {0};
                for (int j = 0 ; j < INPUT_DIMENSION ; j++) {
                    LtNoise[j] = (double) (rand() / (double) RAND_MAX);
                }

                for (int j = 0;j < INPUT_DIMENSION;j++) {
                    LtNoise[j] = LtNoise[j] / ((INPUT_DIMENSION * NormTwo(LtNoise, INPUT_DIMENSION)) / 1.0) ;
                    inputVector[j] += LtNoise[j];
                    inputTSum[j] += inputVector[j];
                    minuend += weightVector[j] * inputVector[j];
                }
                double substractor = MAXFLOAT;
                for (int i = 0;i < INPUT_DIMENSION;i++) {
                    if (inputTSum[i] < substractor)
                        substractor = inputTSum[i];
                }
                currentRegret += minuend - substractor;

                // Check Mark
                double upperPart[INPUT_DIMENSION];
                upperPart[INPUT_DIMENSION] = {0};
                for (int i = 0;i < INPUT_DIMENSION;i++) {
                    upperPart[i] = exp(inputTSum[i] * (-1) * eta);
                }
                double downPartNumber = 0;
                for (int i = 0;i < INPUT_DIMENSION;i++)
                    downPartNumber += upperPart[i];

                // Renew weight
                for (int i = 0;i < INPUT_DIMENSION;i++) {
                    weightVector[i] = upperPart[i] / (downPartNumber / 1.0);
                }

            }
            regret[regretI] = currentRegret;
            regretI++;
    }

    for (int i = 0; i< INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER;i++) cout << regret[i] << endl;
    double baseline[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    baseline[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER] = {0};
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
    cout << "Hedge Done" << endl;
//    return regretRe;
}

void HedgeMain(int INPUT_DIMENSION_LOWER, int  INPUT_DIMENSION_UPPER, int ROUND) {
    HedgeAlgorithm(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, ROUND);
}