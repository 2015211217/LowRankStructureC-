//
// Created by holmes on 2021/8/20.
//

# include "LowRankExpertAdvice.h"
# include <iostream>
# include <cmath>
# include "MVEE.h"

#include "MirrorDescend.h"
#include <Eigen/Eigenvalues>
#include "gurobi_c++.h"
#include <iostream>
#include "SchimidtOrth.h"
#include <limits.h>

using namespace std;
using namespace Eigen;

MatrixXd LowRankExpertAdvice(MatrixXd data, int DIMENSION, int ROUND, double EPSILON) {
    MatrixXd regret, inputAccumulation, weightVector, Lt, Vk, M;
    regret.resize(1, ROUND);
    Lt.resize(1, DIMENSION);
    Vk.resize(DIMENSION, 1);
    inputAccumulation.resize(1, DIMENSION);
    weightVector.resize(1, DIMENSION);
    inputAccumulation.fill(0);
    weightVector.fill(1/(DIMENSION / 1.0));
    double currentLossM = 0;
    double tau = 0;
    int regretI = 0;
    bool rankFlag = true;
    GRBEnv env = GRBEnv();
    env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_ScaleFlag, 0);
    for (int t = 1; t <= ROUND;t++) {
        for (int i = 0;i < DIMENSION;i++) {
            Lt(0, i) = data(t - 1, i);
            inputAccumulation(0, i) += Lt(0, i);
            currentLossM += Lt(0, i) * weightVector(0, i);
        }
        regret(0, regretI) = currentLossM - inputAccumulation.minCoeff();
        regretI++;
        bool flag = true;
        FullPivLU<MatrixXd> Decomp(Vk.transpose());
        MatrixXd newVk = Vk;
        newVk.conservativeResize(DIMENSION, Vk.cols() + 1);
        for (int i = 0;i < DIMENSION;i++)
            newVk(i, Vk.cols()) = Lt(0, i); // Vk N * cols
        FullPivLU<MatrixXd> Comp(newVk.transpose());
        if (Decomp.rank() == DIMENSION) rankFlag = false;
        if (t == 1) {
            flag = false;
            Vk = Lt.transpose(); // Vk is N * a
        } else if(rankFlag and Comp.rank() > Decomp.rank()) {
            Vk = newVk;
            flag = false;
        }
        if (!flag) {
            tau = 0;
            MatrixXd MVEEInput = Vk;
            MVEEInput.conservativeResize(DIMENSION * 2, Vk.cols());
            for (int i = 0;i < Vk.rows();i++)
                for (int j = 0;j < Vk.cols();j++)
                    MVEEInput(DIMENSION + i, j) = (-1) * Vk(i, j);
            M = MVEE(DIMENSION, Vk.cols(), MVEEInput, EPSILON);

        }
        tau += 1;
        FullPivLU<MatrixXd> compVk(Vk);
        double eta = 4 * sqrt(compVk.rank() / (tau / 1.0));

        try {
            GRBModel model = GRBModel(env);
            // create variables
            GRBVar weightGRB[DIMENSION];
            MatrixXd At;
            MatrixXd Identity;
            for (int i = 0; i < DIMENSION; i++)
                weightGRB[i] = model.addVar(0.0, 1.0, 1, GRB_CONTINUOUS);
            // target function creation
            GRBQuadExpr obj;
            for (int i = 0; i < DIMENSION; i++)
                obj += weightGRB[i] * Lt(0, i);
            At = Identity.setIdentity(DIMENSION, DIMENSION) + Vk * (M * Vk.transpose());

            for (int i = 0; i < DIMENSION; i++)
                for (int j = 0;j < DIMENSION;j++)
                    obj += (1 / (eta / 1.0)) * (weightGRB[i] - weightVector(0, i)) * At(j, i) * (weightGRB[i] - weightVector(0, i));
            GRBLinExpr lhs = 0;
            // sum is one
            for (int i = 0; i < DIMENSION; i++)
                lhs += weightGRB[i];
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
