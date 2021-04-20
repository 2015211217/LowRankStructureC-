//
// Created by holmes on 2021/3/29.
//
#ifndef LOWRANKDESCENDC___ONLINEPCA_H
#define LOWRANKDESCENDC___ONLINEPCA_H
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



#endif //LOWRANKDESCENDC___ONLINEPCA_H
