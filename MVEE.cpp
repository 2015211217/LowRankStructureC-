//
// Created by holmes on 2021/3/29.
//
#include <iostream>
//#include "math.h"
#define ld long double
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#define gamma 1
using namespace std;
using namespace Eigen;

Matrix<long double, -1, -1> MVEE(const int INPUT_DIMENSION, const int INPUT_RANK, Matrix<long double, -1, -1>Vm) {
    Matrix<ld, Dynamic, Dynamic> Ek;
    for (int j = 0;j < INPUT_RANK;j++) {
        for (int i = 0; i < INPUT_RANK; i++) {
            if (i == j)
                Ek(i,i) = INPUT_RANK * sqrt(INPUT_DIMENSION);
            else Ek(i,j) = 0;
        }
    }
    for (int i = 0;i < INPUT_DIMENSION * 2 ;i++) {
        if ((Vm.col(i) * Ek * Vm.col(i).transpose()).sqrt() == 0) continue;
        if (pow(INPUT_RANK + 1, -(1/2)) * Vm.col(i) * Ek * Vm.col(i).transpose() >= 1) {
            ld alpha = (-1) * gamma / sqrt(Vm.col(i) * Ek * Vm.col(i).transpose());
            if (alpha < (-1) / sqrt(INPUT_RANK)) continue;
            else if (alpha < 0 && alpha >= (-1) / sqrt(INPUT_RANK)) {
                Matrix<ld, Dynamic, Dynamic> b;
                for (int k = 0;k < INPUT_RANK ; k++) {
                    b = Ek * Vm.col(i).transpose() / sqrt(Vm.col(i) * Ek * Vm.col(i).transpose());
                }
//                for (int ik = 0; ik < INPUT_RANK ; ik++)
//                    for (int jk = 0; jk < INPUT_RANK ; jk++) {
//
//                    }
                Ek = (INPUT_RANK / (INPUT_RANK - 1)) * (1 - pow(alpha, 2)) * (Ek - b * b.transpose() * (1 - INPUT_RANK * pow(alpha, 2)) / (1 - pow(alpha, 2)));
            } else cout << "ERROR FOR MVEE PRECESSION !!" <<endl;
        }

    }
    return Ek;
}