//
// Created by holmes on 2021/4/1.
//
#include <iostream>
#include "HedgeAlgorithm.h"
#include "MirrorDescend.h"
#define INPUT_DIMENSION_UPPER 200
#define INPUT_DIMENSION_LOWER 200
#define INPUT_RANK 2
#define ROUND 10
#define KAISHU 10
using namespace std;

int main() {
//    HedgeMain(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, ROUND);
    MirrorDescendMain(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, INPUT_RANK, ROUND, KAISHU);
    return 0;
}
