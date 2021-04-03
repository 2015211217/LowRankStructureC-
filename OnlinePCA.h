//
// Created by holmes on 2021/3/29.
//

#ifndef LOWRANKDESCENDC___ONLINEPCA_H
#define LOWRANKDESCENDC___ONLINEPCA_H
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "OnlinePCA.h"
using namespace std;
using namespace Eigen;

OnlinePCAReturn OnlinePCA(int INPUT_DIMENSION, int INPUT_RANK, double eta, double alpha, MatrixXd Lt, MatrixXd w_last, double AccumulatePCA, MatrixXd PLast);

struct OnlinePCAReturn {
    MatrixXd P;
    MatrixXd W;
    double AccumulatePCA;
    MatrixXd PLast;
}PCAReturn;

#endif //LOWRANKDESCENDC___ONLINEPCA_H
