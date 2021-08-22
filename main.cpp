//
// Created by holmes on 2021/4/1.
//
#include <iostream>
#include "HedgeAlgorithm.h"
//#include "MirrorDescend.h"
#include "Algorithm512.h"
#include "LowRankExpertAdvice.h"
#include "Eigen/Core"
#include <Eigen/Eigenvalues>
#include "random"
#include "SchimidtOrth.h"

#define INPUT_DIMENSION 200
#define INPUT_RANK 2
#define ROUND 1000
#define KAISHU 5
#define EPSILON 0
using namespace std;
using namespace Eigen;

int main() {
    // INPUT CREATION
    MatrixXd INPUT;
    INPUT.resize(ROUND * KAISHU, INPUT_DIMENSION);
    random_device rd;
    default_random_engine eng(rd());
    uniform_real_distribution<> distr(-1, 1);
    MatrixXd U;

    while(1) {
        U.resize(INPUT_DIMENSION, INPUT_RANK);
        for (int i = 0;i < INPUT_DIMENSION;i++)
            for(int j = 0;j < INPUT_RANK;j++)
                U (i, j) =  distr(eng);

        FullPivLU<MatrixXd>  Decomp(U);
        if (Decomp.rank() == INPUT_RANK) break;
    }

    for (int t = 0; t < KAISHU * ROUND; t++) {
        MatrixXd V, Lt, LtNoise;
        V.resize(1, INPUT_RANK);
        Lt.resize(INPUT_DIMENSION, 1);
        for (int i = 0;i < INPUT_RANK;i++)
            V(0, i) = distr(eng);

        Lt = MatrixXd(U * V.transpose()); // N * 1
        LtNoise = MatrixXd(1, INPUT_DIMENSION); // 1 * N
        for (int i = 0;i < INPUT_DIMENSION;i++) {
            Lt(i, 0) = Lt(i, 0) / (1.0 * Lt.maxCoeff());
            LtNoise(0, i) = distr(eng);
        }
        if (EPSILON != 0) {
            LtNoise = LtNoise / (1.0 * (1 / (EPSILON * 0.1)) * NormTwo(LtNoise, INPUT_DIMENSION));
            Lt = Lt + LtNoise.transpose();
        }
        for (int i = 0;i < INPUT_DIMENSION;i++)
            INPUT(t, i) = Lt(i, 0);
    }

    MatrixXd HedgeR, LowRankR, R512;
    HedgeR.resize(KAISHU, ROUND);
    LowRankR.resize(KAISHU, ROUND);
    R512.resize(KAISHU, ROUND);
    for (int i = 0;i < KAISHU;i++) {
        cout << i << endl;
        MatrixXd inputLine;
        inputLine.resize(ROUND, INPUT_DIMENSION);
        for (int j = 0; j < ROUND; j++)
            for (int k = 0;k < INPUT_DIMENSION;k++)
                inputLine(j, k) = INPUT(ROUND * i + j, k);
        MatrixXd HedgeRegret = HedgeMain(inputLine, INPUT_DIMENSION, INPUT_RANK, ROUND);
        cout << HedgeRegret << endl;
        MatrixXd Regret512 = Algorithm512(inputLine, INPUT_DIMENSION, INPUT_RANK, ROUND, EPSILON);
        cout << Regret512 << endl;
        MatrixXd LowRankRegret = LowRankExpertAdvice(inputLine, INPUT_DIMENSION, ROUND, EPSILON);
        cout << LowRankRegret << endl;
        for (int j = 0;j < ROUND;j++) {
            HedgeR(i, j) = HedgeRegret(0, j);
            LowRankR(i, j) = LowRankRegret(0, j);
            R512(i, j) = Regret512(0, j);
        }
    }

    // Find the largest member for each line
    for (int i = 0;i < ROUND;i++) {
        double HedgeMax = -10000;
        double LowRankMax = -10000;
        double Max512 = -10000;
        for (int j = 0; j < KAISHU; j++) {
            if (HedgeMax < HedgeR(j, i)) HedgeMax = HedgeR(j, i);
            if (LowRankMax < LowRankR(j, i)) LowRankMax = LowRankR(j, i);
            if (Max512 < R512(j, i)) Max512 = R512(j, i);
        }
        HedgeR(0, i) = HedgeMax;
        LowRankR(0, i) = LowRankMax;
        R512(0, i) = Max512;
    }
    for (int i = 0;i < ROUND; i++)
        cout << HedgeR(0, i) << " ";
    cout << endl;
    for (int i = 0;i < ROUND; i++)
        cout << LowRankR(0, i) <<" ";
    cout << endl;
    for (int i = 0;i < ROUND; i++)
        cout << R512(0, i) <<" ";
    cout << endl;
    return 0;
}
