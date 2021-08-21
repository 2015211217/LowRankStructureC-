//
// Created by holmes on 2021/3/29.
//

#ifndef LOWRANKDESCENDC___MVEE_H
#define LOWRANKDESCENDC___MVEE_H
#include "Eigen/Core"
#include "Eigen/Dense"
using namespace Eigen;

MatrixXd MVEE(const int INPUT_DIMENSION, const int INPUT_RANK, MatrixXd Vm, double EPSILON);

#endif //LOWRANKDESCENDC___MVEE_H
