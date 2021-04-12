// Created by holmes on 2021/3/29.
//
# include <iostream>
# include <cmath>
# include "MVEE.h"
# include "OnlinePCA.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MirrorDescend.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "gurobi_c++.h"
//#include "matplotlibcpp.h"

using namespace std;
using namespace cv;
using namespace Eigen;
//using namespace matplotlibcpp;
//WITHOUT_NUMPY = true;

double NormTwo(MatrixXd inputArray, int inputLength) { // Norm calculate
    long double norm = 0;
    for (int j = 0;j < inputLength;j++)
        norm += pow(abs(inputArray(0, j)), 2);
    return pow(norm, 0.5);
}

MatrixXd MirrorDescend(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    Matrix<double, 1, Dynamic> regret;
    regret.conservativeResize(1, INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER);
    Matrix<double, 1, Dynamic> regretBound;
    Matrix<double, Dynamic, Dynamic> inputAccumulation;
    double currentLossM = 0;
    MatrixXd M, V, P;
    MatrixXd PreviousM, PreviousV;

    for (int INPUT_DIMENSION = INPUT_DIMENSION_LOWER; INPUT_DIMENSION < INPUT_DIMENSION_UPPER + 1 ;INPUT_DIMENSION++) {
        cout << INPUT_DIMENSION << endl;
        MatrixXd WLast;
        MatrixXd PLast;
        double AccumulatePCA = 0.0;
        // INPUT GENERATION
        MatrixXd weightVector;
//      WLast = (1. * (INPUT_DIMENSION - INPUT_RANK) / INPUT_DIMENSION) * np.eye(INPUT_DIMENSION)
        WLast.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
        PLast.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);

        for (int i = 0;i < INPUT_DIMENSION;i++)
            for (int j = 0;j < INPUT_DIMENSION;j++) {
                if (i == j) {
                    WLast(i, j) = ((INPUT_DIMENSION - INPUT_RANK) / (INPUT_DIMENSION*1.0));
                    PLast(i, j) = (INPUT_RANK / (INPUT_DIMENSION*1.0));
                }
                else {
                    WLast(i, j) = 0;
                    PLast(i, j) = 0;
                }
            }

        weightVector.conservativeResize(1, INPUT_DIMENSION);
        for (int i = 0; i < INPUT_DIMENSION;i++)
            weightVector(0, i) = 1 / (INPUT_DIMENSION/1.0);
        MatrixXd INPUT_MATRIX;
        INPUT_MATRIX.conservativeResize(INPUT_DIMENSION, INPUT_RANK);
        double currentLossM = 0.0;
        while(1) {
            for (int j = 0;j < INPUT_DIMENSION;j++)
                for (int i = 0;i < INPUT_RANK;i++)
                    INPUT_MATRIX(j, i) = (-1) + (double) ((rand() / (double) RAND_MAX) * 2);
            // Rank calculation
            FullPivLU<MatrixXd> Decomp(INPUT_MATRIX);
            if (Decomp.rank() == INPUT_RANK) break;
        }

        double eta = sqrt(1 / ROUND);
        double AccumulatePartTwo = 0.0;
        MatrixXd eigenValue;
        MatrixXd eigenVector;

            MatrixXd PT1;
            PT1.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
            for (int i = 0;i < INPUT_DIMENSION;i++)
                for (int j = 0; j < INPUT_DIMENSION;j++)
                    if (i >= INPUT_DIMENSION-INPUT_RANK and i == j) PT1(i,j) = 1;
                    else PT1(i,j) = 0;
            EigenSolver<MatrixXd> PSolver1(PT1);
            eigenValue =  (PSolver1.eigenvalues().real()).transpose();
            eigenVector = PSolver1.eigenvectors().real();
            int count1 = 0;
            V.conservativeResize(INPUT_DIMENSION, INPUT_RANK);
            for(int i = 0; i < INPUT_DIMENSION;i++) {
                if (eigenValue(0, i) > 0) {
                    V.col(count1) = eigenVector.col(i);
                    count1++;
                }
            }

            MatrixXd VMVEE;
            VMVEE.conservativeResize(INPUT_DIMENSION * 2, count1);
            for(int i = 0;i < count1;i++) {
                VMVEE.row(i) = V.row(i);
                VMVEE.row(count1 + i) = (-1) * V.row(i);
            }
            M.conservativeResize(INPUT_RANK, INPUT_RANK);
            M = MVEE(INPUT_DIMENSION, INPUT_RANK, VMVEE);
//            for (int i = 0; i < INPUT_DIMENSION;i++)
//                V.row(i) = VMVEE.row(i);

        for (int T = 1 ; T < ROUND + 1 ; T++) {
            // Generate the input
            MatrixXd INPUT_V;
            INPUT_V.conservativeResize(1, INPUT_RANK);
            for (int i = 0; i < INPUT_RANK; i++)
                INPUT_V(0, i) = (-1) + (double) ((rand() / (double) RAND_MAX) * 2);
            Matrix<double, 1, Dynamic> Lt;

            Lt.conservativeResize(1, INPUT_DIMENSION);
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
            LtNoise.conservativeResize(1, INPUT_DIMENSION);
            for (int j = 0; j < INPUT_DIMENSION; j++)
                LtNoise(0, j) = (double) (rand() / (double) RAND_MAX);
            for (int j = 0; j < INPUT_DIMENSION; j++) {
                LtNoise(0, j) /= ((INPUT_DIMENSION * NormTwo(LtNoise, INPUT_DIMENSION)) * 1.0);
                Lt(0, j) += LtNoise(0, j);
                if (Lt(0, j) < 1e-8) Lt(0, j) = 0;
            }



            // Regret and Regret bound
            if (T == 1)
                inputAccumulation = Lt;
            else
                for(int i = 0;i < INPUT_DIMENSION;i++)
                    inputAccumulation(0, i) = inputAccumulation(0, i) + Lt(0, i);
            for(int i = 0; i < weightVector.cols(); i++)
                currentLossM += weightVector(0, i) * Lt.transpose()(i, 0);
            regret(0, INPUT_DIMENSION - INPUT_DIMENSION_LOWER) += currentLossM - inputAccumulation.minCoeff();
            cout<<regret(0, INPUT_DIMENSION - INPUT_DIMENSION_LOWER) << endl;

            try {
                GRBEnv env = GRBEnv();
                env.set(GRB_IntParam_OutputFlag, 0);

                GRBModel model = GRBModel(env);
                // create variables
                GRBVar weightGRB[INPUT_DIMENSION];
                for (int i = 0; i < INPUT_DIMENSION; i++)
                    weightGRB[i] = model.addVar(0.0, 1.0, 1, GRB_CONTINUOUS);
                GRBQuadExpr obj;

                for (int i = 0; i < INPUT_DIMENSION; i++)
                    obj += weightGRB[i] * inputAccumulation(0, i);
                MatrixXd At;
                MatrixXd Identity;

                for(int i  = 0; i < V.rows();i++)
                    for (int j = 0;j < V.cols();j++)
                        if (V(i, j) < 1e-8) V(i, j) = 0;
                for(int i  = 0; i < M.rows();i++)
                    for (int j = 0;j < M.cols();j++)
                        if (M(i, j) < 1e-8) M(i, j) = 0;

                At = Identity.setIdentity(INPUT_DIMENSION, INPUT_DIMENSION) + V * (M * V.transpose());
                for (int i = 0; i < INPUT_DIMENSION; i++)
                    for (int j = 0; j < INPUT_DIMENSION; j++) {
                        obj += weightGRB[i] * weightGRB[j] * At(i, j);
                    }
                GRBLinExpr lhs = 0;
                for (int i = 0; i < INPUT_DIMENSION; i++)
                    lhs += weightGRB[i];
                model.addConstr(lhs == 1, "c0");
                model.setObjective(obj, GRB_MINIMIZE);
                model.optimize();
                for (int i = 0;i < INPUT_DIMENSION;i++)
                    weightVector(0, i) = weightGRB[i].get(GRB_DoubleAttr_X);
            } catch (GRBException e) {
                cout << " Error number : " << e.getErrorCode() << endl;
                cout << e.getMessage() << endl;
            }

            // OnlinePCA and MVEE
//            double eta_adaptive = log(1 + sqrt(2 * log(INPUT_DIMENSION / (T* 1.0)))); //peipeipei

            double alpha = 0.000;
            OnlinePCAReturn PCAReturn;


            PCAReturn = OnlinePCA(INPUT_DIMENSION, INPUT_RANK, eta, alpha, Lt, WLast, AccumulatePCA, PLast);

            P = PCAReturn.Preturn;
            PLast = PCAReturn.PLast;
            AccumulatePCA = PCAReturn.AccumulatePCA;
            WLast = PCAReturn.W;
            for(int i  = 0; i < WLast.rows();i++)
                for (int j = 0;j < WLast.cols();j++)
                    if (WLast(i, j) < 1e-8) WLast(i, j) = 0;
            for(int i  = 0; i < P.rows();i++)
                for (int j = 0;j < P.cols();j++) {
                    if (P(i, j) < 1e-8) P(i, j) = 0;
                    if (PLast(i, j) < 1e-8) PLast(i, j) = 0;
                }


            EigenSolver<MatrixXd> PSolver(P);
            eigenValue =  (PSolver.eigenvalues().real()).transpose();
            eigenVector = PSolver.eigenvectors().real();
            int lineCount = 1;
            for(int i = 0;i < INPUT_DIMENSION;i++) {
                if (eigenValue(0, i) > 1e-5) {
                    V.conservativeResize(INPUT_DIMENSION, lineCount);
                    V.col(lineCount - 1) = eigenVector.col(i);
                    lineCount++;
                }
            }

            lineCount--;
            VMVEE.conservativeResize(INPUT_DIMENSION * 2, lineCount);
            for(int i = 0;i < INPUT_DIMENSION;i++) {
                VMVEE.row(i) = V.row(i);
                for(int j = 0;j < lineCount ; j++)
                    VMVEE(INPUT_DIMENSION + i, j) = (-1) * V(i, j);
            }
            M = MVEE(INPUT_DIMENSION, INPUT_RANK, VMVEE);
        }
    }
    return regret;
}

void MirrorDescendMain(int INPUT_DIMENSION_LOWER, int INPUT_DIMENSION_UPPER, int INPUT_RANK, int ROUND) {
    MatrixXd regret = MirrorDescend(INPUT_DIMENSION_LOWER, INPUT_DIMENSION_UPPER, INPUT_RANK, ROUND);
//    vector<double> baseline;
//    vector<double> TIME_ROUND;
//
//    for (int i = 0; i < (INPUT_DIMENSION_UPPER - INPUT_DIMENSION_LOWER); i++) {
//        baseline.push_back(i + 3);
//        TIME_ROUND.push_back(i+1);
//    }
//    vector<double> regretVector;
//    for(int i = 0;i < regret.cols();i++)
//        regretVector.push_back(regret(0, i));
//    figure_size(1200, 780);
//    plot(TIME_ROUND, baseline, "r--");
//    plot(TIME_ROUND, regretVector, "b--");
////    named_plot("MirrorDescend N");
//    title("MirrorDescend N");
//    legend();
//    save("./NPictureC++.png");
//    cout << "Done" << endl;
}