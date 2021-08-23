//
// Created by holmes on 2021/8/21.
//

#include "SchimidtOrth.h"
#include "iostream"
using namespace Eigen;
using namespace std;

double NormTwo(MatrixXd inputArray, int inputLength) { // Norm calculate
    long double norm = 0;
    for (int j = 0;j < inputLength;j++)
        norm += pow(abs(inputArray(0, j)), 2);
    return pow(norm, 0.5);
}

MatrixXd SchimidtOrth(MatrixXd input) { // Uk is a x N
    MatrixXd result;
    result = input;
    for (int t = 0 ;t < input.rows();t++) {
        MatrixXd currentLine;
        currentLine.resize(1, input.cols());
        currentLine.row(0) = result.row(t);

        for (int i = 0; i < t; i++) {
            for (int j = 0; j < input.cols();j++) {
                currentLine(0, j) -= result(i, j) * ((currentLine(0, j) * result(i, j)) / (1.0 * result(i, j) * result(i, j)));
            }
        }
        result.row(t) = currentLine;
    }

    // guiyihua
    for (int i = 0 ;i < result.rows();i++) {
        double lineSum = result.row(i).sum();
        for (int j = 0;j < result.cols();j++)
            result(i, j) /= (1.0 * lineSum);
    }

    return result;
}
