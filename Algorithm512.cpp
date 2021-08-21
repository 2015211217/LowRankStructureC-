//
// Created by holmes on 2021/8/20.
//

# include "Algorithm512.h"
# include <iostream>
# include <cmath>
# include "MVEE.h"
# include "SchimidtOrth.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include "gurobi_c++.h"
#include <iostream>
using namespace std;
using namespace Eigen;

MatrixXd Algorithm512(MatrixXd data, int DIMENSION, int INPUT_RANK, int ROUND, double epsilon){
    MatrixXd inputAccumulation, regret, weightVector, Lt, Uk, Hk;
    regret.resize(1, ROUND);
    inputAccumulation.resize(1, DIMENSION);
    inputAccumulation.fill(0);
    Lt.resize(1, DIMENSION);
    weightVector.resize(1, DIMENSION);
    weightVector.fill( 1 / (1.0 * DIMENSION));
    double sk, accumulativeWeightInput = 0, gamma;
    int regretI = 0;
    MatrixXd PUk, PUkLt;
    PUk.resize(DIMENSION, 1);
    PUkLt.resize(DIMENSION, 1);
    PUk.fill(0);
    PUkLt.fill(0);
    Uk.resize(1, DIMENSION);
    int k = 0, tau = 0;
    double eta, mk;
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_ScaleFlag, 0);
    Hk = Hk.setIdentity(DIMENSION, DIMENSION);

    for(int t = 1 ; t <= ROUND ; t++) {
        sk = pow(20 * k * DIMENSION * epsilon, 1 / (0.1 * 3));
        for (int i = 0;i < DIMENSION;i++) {
            Lt(0, i) = data(t - 1, i);
            inputAccumulation(0, i) += Lt(0, i);
            accumulativeWeightInput += weightVector(0, i) * Lt(0, i);
        }

        regret(0, regretI) = accumulativeWeightInput - inputAccumulation.minCoeff();
        regretI++;
        // JUDGE
        if (sk == 0) gamma = 0;
        else gamma = sqrt(20 * k * (epsilon / (1.0 * sk)));

        if (sk == 0 && NormTwo(Lt, DIMENSION) < 0 &&
            NormTwo(MatrixXd(Lt - Lt * PUk * PUkLt.transpose()), DIMENSION) <= 2 * epsilon) {
            if (Uk.rows() != INPUT_RANK) {
                Uk = Lt;
                PUk.fill(0);
            }
            else {
                // a * N
                Uk.conservativeResize(Uk.rows() + 1, DIMENSION);
                Uk.row(Uk.rows() - 1) = Lt.row(0);
                // shimite zhengjiaohua
                PUk = SchimidtOrth(Uk);
                if (PUk.size()) PUkLt = Lt * PUk * PUk.transpose();
            }
            if (k < INPUT_RANK) k++;
            MatrixXd m, UkMVEE;
            // N needs to be the row
            UkMVEE = Uk.transpose();
            UkMVEE.conservativeResize(DIMENSION * 2, Uk.rows());
            for (int i = 0;i < DIMENSION;i++)
                for (int j = 0;j < Uk.rows();j++)
                    UkMVEE(DIMENSION + i, j) = Uk(j, i);
            m = MVEE(DIMENSION, Uk.rows(), UkMVEE.transpose(), epsilon);
            Hk = Uk.transpose() * m * Uk;
            for (int i = 0;i < Hk.rows();i++)
                for (int j = 0;j < Hk.cols();j++)
                    if (i == j) Hk(i, j) += 1;
        }
        if (sk > 0) {
            if (NormTwo(Lt, DIMENSION) >= 2 * sk &&
            NormTwo(Lt - Lt * PUk * PUk.transpose(), DIMENSION) > 2 * epsilon + gamma * (NormTwo(Lt, DIMENSION) + epsilon)) {
                if (Uk.rows() != INPUT_RANK) {
                    tau = 0;
                    Uk.conservativeResize(Uk.rows() + 1, DIMENSION);
                    Uk.row(Uk.rows() - 1) = Lt.row(0);
                    if (Uk.size() == 0) PUk.fill(0);
                    else {
                        PUk = SchimidtOrth(Uk);
                        if(PUk.size()) PUkLt = Lt * PUk * PUk.transpose();
                    }
                    if (k < INPUT_RANK) k++;
                    MatrixXd m, UkMVEE;
                    // N needs to be the row
                    UkMVEE = Uk.transpose();
                    UkMVEE.conservativeResize(DIMENSION * 2, Uk.rows());
                    for (int i = 0;i < DIMENSION;i++)
                        for (int j = 0;j < Uk.rows();j++)
                            UkMVEE(DIMENSION + i, j) = Uk(j, i);
                    m = MVEE(DIMENSION, Uk.rows(), UkMVEE.transpose(), epsilon);
                    Hk = Uk.transpose() * m * Uk;
                    for (int i = 0;i < Hk.rows();i++)
                        for (int j = 0;j < Hk.cols();j++)
                            if (i == j) Hk(i, j) += 1;
                }
            }
        }
        tau++;
        if (2 * sk > 6 * epsilon + gamma * sqrt(DIMENSION)) mk = 2 * sk;
        else mk = 6 * epsilon + gamma * sqrt(DIMENSION);
        eta = sqrt((16 * k) / (1.0 * pow(1 + mk,2) * tau));
        // Renew Weight
        try {
            GRBModel model = GRBModel(env);
            // create variables
            GRBVar weightGRB[DIMENSION];
            MatrixXd At;
            MatrixXd Identity;
            for (int i = 0; i < DIMENSION; i++)
                weightGRB[i] = model.addVar(0.0, 1.0, 1, GRB_CONTINUOUS);
            GRBLinExpr lhs = 0;
            // sum is one
            for (int i = 0; i < DIMENSION; i++)
                lhs += weightGRB[i];
            // target function creation
            GRBQuadExpr obj;
            for (int i = 0; i < DIMENSION; i++)
                obj += weightGRB[i] * Lt(0, i);
            for (int i = 0; i < DIMENSION; i++)
                for (int j = 0; j < DIMENSION; j++)
                    obj += pow(eta, -1) * (weightGRB[i] - weightVector(0, i)) * Hk(j, i) * (weightGRB[i] - weightVector(0, i));
            model.addConstr(lhs == 1, "c0");
            model.setObjective(obj, GRB_MINIMIZE);
            model.optimize();
            for (int i = 0;i < DIMENSION;i++)
                weightVector(0, i) = weightGRB[i].get(GRB_DoubleAttr_X);
        } catch (GRBException e) {
            cout << " Error number : " << e.getErrorCode() << endl;
            cout << e.getMessage() << endl;
        }
    }
    return regret;
}
