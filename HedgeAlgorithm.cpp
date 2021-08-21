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
#include "SchimidtOrth.h"

using namespace std;
using namespace cv;
using namespace Eigen;

MatrixXd HedgeMain(MatrixXd data, int INPUT_DIMENSION, int INPUT_RANK, int ROUND) {
    MatrixXd regret;
    regret.resize(1, ROUND);
    int regretI = 0;
    for (int t = 1 ; t <= ROUND ; t++) {
        double weightVector[INPUT_DIMENSION];
        for (int i = 0; i < INPUT_DIMENSION; i++) {
            weightVector[i] = 1 / (INPUT_DIMENSION / 1.0);
        }
        double currentRegret = 0.0;
        double inputTSum[INPUT_DIMENSION];
        inputTSum[INPUT_DIMENSION] = {0};
        double minuend = 0.0;
        double eta = pow(log(INPUT_DIMENSION) / (t / 1.0), 0.5);
        double inputVector[INPUT_DIMENSION];
        for (int i = 0; i < INPUT_DIMENSION; i++)
            inputVector[i] = data(t - 1, i);
        for (int j = 0; j < INPUT_DIMENSION; j++) {
            inputTSum[j] += inputVector[j];
            minuend += weightVector[j] * inputVector[j];
        }
        double substractor = MAXFLOAT;
        for (int i = 0; i < INPUT_DIMENSION; i++) {
            if (inputTSum[i] < substractor)
                substractor = inputTSum[i];
        }
        currentRegret += minuend - substractor;
        double upperPart[INPUT_DIMENSION];
        upperPart[INPUT_DIMENSION] = {0};
        for (int i = 0; i < INPUT_DIMENSION; i++) {
            upperPart[i] = exp(inputTSum[i] * (-1) * eta);
        }
        double downPartNumber = 0;
        for (int i = 0; i < INPUT_DIMENSION; i++)
            downPartNumber += upperPart[i];
        // Renew weight
        for (int i = 0; i < INPUT_DIMENSION; i++) {
            weightVector[i] = upperPart[i] / (downPartNumber / 1.0);
        }
        regret(0, regretI) = currentRegret;
        regretI++;
    }
    return regret;
}
