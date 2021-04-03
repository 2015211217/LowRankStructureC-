//
// Created by holmes on 2021/4/1.
//
#include <iostream>
#include "HedgeAlgorithm.h"
#include "MirrorDescend.h"
#define INPUT_DIMENSION_UPPER 100
#define INPUT_DIMENSION_LOWER 4
#define INPUT_RANK 3
#define ROUND 100
using namespace std;

int main() {
//    HedgeMain(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, ROUND);
    MirrorDescendMain(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, INPUT_RANK, ROUND);
    return 0;
}
