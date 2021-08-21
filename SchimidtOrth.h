//
// Created by holmes on 2021/8/21.
//

#ifndef LOWRANKDESCENDC_SCHIMIDTORTH_H
#define LOWRANKDESCENDC_SCHIMIDTORTH_H
#include "Eigen/Core"

Eigen::MatrixXd SchimidtOrth(Eigen::MatrixXd input);
double NormTwo(Eigen::MatrixXd inputArray, int inputLength);


#endif //LOWRANKDESCENDC_SCHIMIDTORTH_H
