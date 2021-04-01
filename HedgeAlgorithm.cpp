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

#define INPUT_DIMENSION_UPPER 100
#define INPUT_DIMENSION_LOWER 4
#define ROUND 10

long double NormTwo(long double *inputArray, int inputLength) { // Norm calculate
    long double norm = 0;
    for (int j = 0;j < inputLength;j++)
        norm += pow(abs(inputArray[j]), 2);
    return pow(norm, 0.5);
}

long double *HedgeAlgorithm() {
    long double regret[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    int regretI = 0;
    srand(time(NULL));
    for (int INPUT_DIMENSION = INPUT_DIMENSION_LOWER ; INPUT_DIMENSION <= INPUT_DIMENSION_UPPER ; INPUT_DIMENSION++) {
        cout << INPUT_DIMENSION <<endl;
        // Init weight vector
        long double weightVector[INPUT_DIMENSION];
        for (int i = 0; i < INPUT_DIMENSION;i++) {
            weightVector[i] = 1 / (INPUT_DIMENSION / 1.0);
        }
        long double currentRegret = 0;
        long double inputTSum[INPUT_DIMENSION];
        inputTSum[INPUT_DIMENSION] = {0};

        for (int T = ROUND; T < ROUND + 1;T++)
            for (int t = 0; t < T ; t++) {
                long double minuend = 0;
                long double eta = pow(log(INPUT_DIMENSION) / (T / 1.0), 0.5);
                long double inputVector[INPUT_DIMENSION];
                inputVector[INPUT_DIMENSION] = {0};
                for (int i = 0;i < INPUT_DIMENSION; i++) {
                    inputVector[i] = (long double) (rand() / (double) RAND_MAX);
                }

                long double LtNoise[INPUT_DIMENSION];
                for (int j = 0 ; j < INPUT_DIMENSION ; j++) {
                    LtNoise[j] = (double) (rand() / (double) RAND_MAX);
                }

                for (int j = 0;j < INPUT_DIMENSION;j++) {
                    LtNoise[j] = LtNoise[j] / ((INPUT_DIMENSION * NormTwo(LtNoise, INPUT_DIMENSION)) / 1.0) ;
                    inputVector[j] += LtNoise[j];
                    inputTSum[j] += inputVector[j];
                    if (inputTSum[j] < 1e-5) inputTSum[j] = 0;
                    cout << inputTSum[j] << endl;

                    minuend += weightVector[j] * inputVector[j];
                }

                // Check Mark
                long double upperPart[INPUT_DIMENSION];
                upperPart[INPUT_DIMENSION] = {0};
                for (int i = 0;i < INPUT_DIMENSION;i++) {
                    upperPart[i] = exp(inputTSum[i] * (-1) * eta);

                }
                long double downPartNumber = 0;
                for (int i = 0;i < INPUT_DIMENSION;i++)
                    downPartNumber += upperPart[i];

                // Renew weight
                for (int i = 0;i < INPUT_DIMENSION;i++) {

                    weightVector[i] = upperPart[i] / (downPartNumber / 1.0);
                }

                long double substractor = MAXFLOAT;
                for (int i = 0;i < INPUT_DIMENSION;i++) {
                    if (inputTSum[i] < substractor)
                        substractor = inputTSum[i];
                }
                currentRegret += minuend - substractor;

            }
            regret[regretI] = currentRegret;
            regretI++;
    }
    return regret;
}

void HedgeMain() {
    long double *regret = HedgeAlgorithm();
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