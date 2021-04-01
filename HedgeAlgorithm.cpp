//
// Created by holmes on 2021/3/29.
//
// N changing version, but it's easy to get the Round one
#include <iostream>
#include "math.h"
#include "random"
#include "time.h"
using namespace std;

#define INPUT_DIMENSION_UPPER 100
#define INPUT_DIMENSION_LOWER 4
#define ROUND 100

int *HedgeAlgorithm() {
    double regret[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    srand(time(NULL));
    for (int INPUT_DIMENSION = INPUT_DIMENSION_LOWER; INPUT_DIMENSION <= INPUT_DIMENSION_UPPER;INPUT_DIMENSION++) {
        cout << INPUT_DIMENSION <<endl;
        // Init weight vector
        double weightVector[INPUT_DIMENSION];
        for (int i = 0; i < INPUT_DIMENSION;i++)
            weightVector[i] = (1 / INPUT_DIMENSION);
        double currentRegret = 0, minuend = 0;
        double inputTSum[INPUT_DIMENSION];
        for (int T = 1; T < ROUND + 1;T++)
            for (int t = 0; t < T ; t++) {
                double eta = pow(log(INPUT_DIMENSION) / T, 0.5);
                double inputVector[INPUT_DIMENSION];
                for (int i = 0;i < INPUT_DIMENSION; i++)
                    inputVector[i] = (-1) + rand() % (1 - (-1));
                double LtNoise[INPUT_DIMENSION];
                for (int j = 0;j < INPUT_DIMENSION;j++)
                    LtNoise[j] = rand() % 1;

                // Check Mark

            }
    }
}

int main() {

    return 0;
}