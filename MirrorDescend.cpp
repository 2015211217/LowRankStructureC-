// Created by holmes on 2021/3/29.
//
# include <iostream>
# include <cmath>
# include "MVEE.h"
# include "OnlinePCA.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MirrorDescend.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include "glpk.h"

using namespace std;
using namespace cv;
using namespace Eigen;

double NormTwo(Matrix<double, -1, -1> inputArray, int inputLength) { // Norm calculate
    long double norm = 0;
    for (int j = 0;j < inputLength;j++)
        norm += pow(abs(inputArray[j]), 2);
    return pow(norm, 0.5);
}

Matrix<double , -1, -1> MirrorDescend(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    Matrix<double, 1, Dynamic> regret;
    Matrix<double, 1, Dynamic> regretBound;
    Matrix<double, Dynamic, Dynamic> inputAccumulation;
    double currentLossM = 0;
    MatrixXd M, V;
    MatrixXd PreviousM, PreviousV;

    for (int INPUT_DIMENSION = INPUT_DIMENSION_LOWER; INPUT_DIMENSION < INPUT_DIMENSION_UPPER + 1 ;INPUT_DIMENSION++) {
        MatrixXd WLast;
        MatrixXd PLast;
        double AccumulatePCA = 0.0;
        // INPUT GENERATION
        Matrix<double, 1, Dynamic> weightVector;
        weightVector.resize(1, INPUT_DIMENSION);
        for (int i = 0; i < INPUT_DIMENSION;i++)
            weightVector(0, i) = 1 / (INPUT_DIMENSION/1.0);
//        cout << weightVector << endl;
        Matrix<double, Dynamic, Dynamic> INPUT_MATRIX;
        INPUT_MATRIX.resize(INPUT_DIMENSION, INPUT_RANK);
        double currentLossM = 0.0;
        while(1) {
            for (int j = 0;j < INPUT_DIMENSION;j++)
                for (int i = 0;i < INPUT_RANK;i++)
                    INPUT_MATRIX(j, i) = (-1) + (double) ((rand() / (double) RAND_MAX) * 2);
            // Rank calculation
            FullPivLU<MatrixXd> Decomp(INPUT_MATRIX);
            if (Decomp.rank() == INPUT_RANK) break;
        }
//        cout << INPUT_MATRIX <<endl;
        double eta = sqrt(1 / ROUND);
        double AccumulatePartTwo = 0.0;
        for (int T = 0 ; T < ROUND ; T++) {
            cout << T;
            // Generate the input
            Matrix<double, Dynamic, Dynamic> INPUT_V;
            INPUT_V.resize(1, INPUT_RANK);
            for (int i = 0; i < INPUT_RANK; i++)
                INPUT_V(0, i) = (-1) + (double) ((rand() / (double) RAND_MAX) * 2);
            Matrix<double, 1, Dynamic> Lt;

            Lt.resize(1, INPUT_DIMENSION);
            for (int i = 0; i < INPUT_DIMENSION; i++)
                for (int j = 0; j < INPUT_RANK; j++)
                    Lt(0, i) += INPUT_MATRIX(i, j) * INPUT_V(0, j);

            // Check Mark
            double biggestNorm = MAXFLOAT * (-1);
            for (int i = 0; i < INPUT_DIMENSION; i++)
                if (biggestNorm < Lt(0, i))
                    biggestNorm = Lt(0, i);
            for (int i = 0; i < INPUT_DIMENSION; i++)
                Lt(0, i) /= biggestNorm;

            Matrix<double, 1, Dynamic> LtNoise;
            LtNoise.resize(1, INPUT_DIMENSION);
            for (int j = 0; j < INPUT_DIMENSION; j++)
                LtNoise(0, j) = (double) (rand() / (double) RAND_MAX);
            for (int j = 0; j < INPUT_DIMENSION; j++) {
                LtNoise(0, j) /= (INPUT_DIMENSION * NormTwo(LtNoise, INPUT_DIMENSION));
                Lt(0, j) += LtNoise(0, j);
            }
            // Regret and Regret bound
            if (T == 1) {
                 inputAccumulation = Lt;
            } else {
                inputAccumulation = inputAccumulation + Lt;
            }
            currentLossM = currentLossM + (double)weightVector.dot(Lt);
            if (T == 1) {
                MatrixXd V;
//                Matrix<double, Dynamic, Dynamic> P;
                for(int i = 0;i < INPUT_DIMENSION;i++)
                    for (int j = 0;j < INPUT_DIMENSION;j++)
                        if(i == j and i >= (INPUT_DIMENSION - INPUT_RANK)) P(i, j) = 0;
                        else P(i, j) = 0;

                        EigenSolver<MatrixXd> PSolver(P);
                        Matrix2d eigenValue =  PSolver.eigenvalues();
                        MatrixXd eigenVector = PSolver.eigenvectors();
                        int lineCount = 0;
                        for(int i = 0;i < INPUT_DIMENSION;i++) {
                            if (eigenValue(i) > 0) {
                                V.row(lineCount) = ((MatrixXd)eigenVector.col(i)).transpose();
                                lineCount++;
                            }
                        }
                    for(int i = 0;i < INPUT_DIMENSION;i++) {
                        if (eigenValue(i) > 0) {
                            V.row(lineCount) = (-1) * ((MatrixXd)eigenVector.col(i)).transpose();
                            lineCount++;
                        }
                    }
                    // Check Mark
                    M = MVEE(INPUT_DIMENSION, INPUT_RANK, V);
                    MatrixXd  weightBest;
                    weightBest.resize(1, INPUT_DIMENSION);
                    weightBest.fill(0);
                    int argminBest = 0;
                    for (int i = 0; i < INPUT_DIMENSION;i++)
                        if(inputAccumulation(0, argminBest) < inputAccumulation(0, i)) argminBest = i;
                        weightBest(0, argminBest) = 1;
                    double PartOne;
                    double PartTwo;
                    double PartThree;
                    double PartFour;
                    MatrixXd Identity;
                    Identity.setIdentity(INPUT_DIMENSION, INPUT_DIMENSION);
                    PartOne = (weightVector * (Identity + V * M * V.transpose()) * weightVector.transpose());
                    PartTwo = Lt * Identity * Lt.transpose();
                    PartThree = weightVector * weightVector.transpose();
                    PartFour = weightVector * (Identity + V * M * V.transpose()) * weightVector.transpose();
                    AccumulatePartTwo += PartOne/(2*eta) + (PartThree - PartFour) / (2 * eta);
                    regretBound(0, INPUT_DIMENSION - INPUT_DIMENSION_LOWER) = PartOne / (2 * eta) + AccumulatePartTwo;
                    regret(0, INPUT_DIMENSION - INPUT_DIMENSION_LOWER) = currentLossM - (double)inputAccumulation.minCoeff();
            } else {
                MatrixXd  weightBest;
                weightBest.resize(1, INPUT_DIMENSION);
                weightBest.fill(0);
                int argminBest = 0;
                for (int i = 0; i < INPUT_DIMENSION;i++)
                    if(inputAccumulation(0, argminBest) < inputAccumulation(0, i)) argminBest = i;
                weightBest(0, argminBest) = 1;
                double PartOne;
                double PartTwo;
                double PartThree;
                double PartFour;
                MatrixXd Identity;
                PartOne = weightBest * (Identity + V * M * V.transpose()) * weightBest.transpose();
                PartTwo = Lt * (Identity + PreviousV * PreviousM * PreviousV.transpose()) * Lt.transpose();
                PartThree = weightVector * (Identity + PreviousV * PreviousM * PreviousV.transpose()) * weightVector;
                PartFour = weightVector * (Identity + V * M * V.transpose()) * weightVector.transpose();
                AccumulatePartTwo += eta * PartTwo + (PartThree - PartFour) / (2 * eta);
                regretBound(0, INPUT_DIMENSION - INPUT_DIMENSION_LOWER) = PartOne / (2 * eta) + AccumulatePartTwo;
                regret(0, INPUT_DIMENSION - INPUT_DIMENSION_LOWER) = currentLossM - (double)inputAccumulation.minCoeff();
            }
            PreviousV = V;
            PreviousM = M;
            // Renew weight
//        fun = lambda x: np.ravel(eta * np.dot(np.transpose(x), inputAccumulation) + np.dot(
//            np.dot(x, np.identity(INPUT_DIMENSION) + np.dot(np.dot(V.real, M), np.transpose(V.real))), np.transpose(x)))
//        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
//        number = []
//        for i in range(INPUT_DIMENSION):
//            number.append(i)
//        a = tuple(map(cons1, number))
//        cons = cons + a
//        OptimizeResult = minimize(fun, weightVector, method='SLSQP', constraints=cons)
//        weightVector = OptimizeResult['x']


            // OnlinePCA and MVEE
            double eta_adaptive = log(1 + sqrt(2 * log(INPUT_DIMENSION / T)));
            double alpha = 0.000;
            OnlinePCAReturn PCAReturn;
            PCAReturn = OnlinePCA(INPUT_DIMENSION, INPUT_RANK, eta, alpha, Lt, WLast, AccumulatePCA, PLast);
            P = PCAReturn.P;
            for(int i = 0;i < INPUT_DIMENSION;i++)
                for (int j = 0;j < INPUT_DIMENSION;j++)
                    if(i == j and i >= (INPUT_DIMENSION - INPUT_RANK)) P(i, j) = 0;
                    else P(i, j) = 0;

            MatrixXd V;
            EigenSolver<MatrixXd> PSolver(PCAReturn.P);
            Matrix2d eigenValue =  PSolver.eigenvalues();
            MatrixXd eigenVector = PSolver.eigenvectors();
            lineCount = 0;
            for(int i = 0;i < INPUT_DIMENSION;i++) {
                if (eigenValue(i) > 0) {
                    V.row(lineCount) = ((MatrixXd)eigenVector.col(i)).transpose();
                    lineCount++;
                }
            }
            for(int i = 0;i < INPUT_DIMENSION;i++) {
                if (eigenValue(i) > 0) {
                    V.row(lineCount) = (-1) * ((MatrixXd)eigenVector.col(i)).transpose();
                    lineCount++;
                }
            }
            M = MVEE(INPUT_DIMENSION, INPUT_RANK, V);
        }
    }
    return regretBound;
}

void MirrorDescendMain(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    Matrix<double, -1, -1> regret = MirrorDescend(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, INPUT_RANK, ROUND);
    double baseline[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    int TIME_ROUND[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];

    for (int i = 0; i < (INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER); i++) {
        baseline[i] = i + 3;
        TIME_ROUND[i] = i + 1;
    }
    Mat img = Mat::zeros(Size(800, 600), CV_8UC3);
    img.setTo(255);
    Point point[INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER];
    for (int i = 0; i < (INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER); i++) {
        Point p(TIME_ROUND[i], regret[i]); point[i] = p;}
    for (int i = 0; i < (INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER) - 1; i++)
        line(img, point[i], point[i+1], Scalar(0, 0, 255), 2);
    imshow("regert - N", img);

    waitKey();
    cout << "Done" << endl;
}