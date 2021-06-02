//
// Created by holmes on 2021/5/31.
//

#ifndef LOWRANKDESCENDC_ONLINEPCANEWVERSION_H
#define LOWRANKDESCENDC_ONLINEPCANEWVERSION_H
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/SVD"
#include <random>
#include "algorithm"

#include "gurobi_c++.h"
using namespace std;
using namespace Eigen;

struct OnlinePCAReturn {
    MatrixXd Preturn;
    MatrixXd W;
    double AccumulatePCA;
    MatrixXd PLast;
};
OnlinePCAReturn OnlinePCA(int INPUT_DIMENSION, int INPUT_RANK, double eta, double alpha, MatrixXd Lt, MatrixXd w_last, double AccumulatePCA, MatrixXd PLast);

#endif //LOWRANKDESCENDC_ONLINEPCANEWVERSION_H
