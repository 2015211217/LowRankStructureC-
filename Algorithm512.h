//
// Created by holmes on 2021/8/20.
//

#ifndef LOWRANKDESCENDC_ALGORITHM512_H
#define LOWRANKDESCENDC_ALGORITHM512_H

#include <Eigen/Core>

Eigen::MatrixXd Algorithm512(Eigen::MatrixXd data,int DIMENSION, int INPUT_RANK, int ROUND, double epsilon);

#endif //LOWRANKDESCENDC_ALGORITHM512_H
