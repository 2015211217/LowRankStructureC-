//
// Created by holmes on 2021/3/29.
//
#include <iostream>
#include <Eigen/Core>

#define gamma 1
using namespace std;
using namespace Eigen;

MatrixXd MVEE(const int INPUT_DIMENSION, const int INPUT_RANK, MatrixXd Vm) {
    MatrixXd Ek;
    Ek.conservativeResize(INPUT_RANK, INPUT_RANK);
    for (int j = 0;j < INPUT_RANK;j++) {
        for (int i = 0; i < INPUT_RANK; i++) {
            if (i == j)
                Ek(i,i) = INPUT_RANK * sqrt(INPUT_DIMENSION);
            else Ek(i,j) = 0;
        }
    }

    for (int i = 0;i < INPUT_DIMENSION * 2 ;i++) {
        double tempOne = sqrt(Vm.row(i) * Ek * Vm.row(i).transpose());
        if (tempOne == 0.0) continue;

        if (pow(INPUT_RANK + 1, -(1/2)) * Vm.row(i) * Ek * Vm.row(i).transpose() >= 1) {
            double alpha = (-1) * gamma / (sqrt(Vm.row(i) * Ek * Vm.row(i).transpose()) * 1.0);
            if (alpha < (-1) / (sqrt(INPUT_RANK)* 1.0)) continue;
            else if (alpha < 0 && alpha >= (-1) / (sqrt(INPUT_RANK)* 1.0)) {
                MatrixXd b;

                for (int k = 0;k < INPUT_RANK ; k++)
                    b = (MatrixXd)((Ek * Vm.row(i).transpose()) / (sqrt(Vm.row(i) * Ek * Vm.row(i).transpose())* 1.0));

                Ek = (INPUT_RANK / ((INPUT_RANK - 1)* 1.0)) * (1 - pow(alpha, 2)) * (Ek - (MatrixXd)(b * b.transpose() * (1 - INPUT_RANK * pow(alpha, 2)) / ((1 - pow(alpha, 2))* 1.0)));

            } else cout << "ERROR FOR MVEE PROCESSION !!" <<endl;
        }
    }

    return Ek;
}