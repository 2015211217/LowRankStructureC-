//
// Created by holmes on 2021/3/29.
//
#include <iostream>
#include "math.h"
#define ld long double
#include <Eigen/Core>
#define gamma 1
using namespace std;
using namespace Eigen;

ld *MVEE(int INPUT_DIMENSION, int INPUT_RANK, ld **Vm) {
    ld Ek[INPUT_RANK][INPUT_RANK];
    Ek[INPUT_RANK][INPUT_RANK] = {0};
    for(int i = 0;i < INPUT_RANK;i++) {
        Ek[i][i] = INPUT_RANK * sqrt(INPUT_DIMENSION);
    }
    for (int i = 0;i < INPUT_DIMENSION * 2 ;i++) {

    }
}