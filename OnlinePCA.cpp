//
// Created by holmes on 2021/3/29.
//

#include "OnlinePCA.h"
#include "Eigen/SVD"
#include <random>
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

void mixture_decompose_list(MatrixXd eignval_W, int INPUT_DIMENSION, int INPUT_RANK, MatrixXd &r_candidate, MatrixXd &p_dist) {
    // start decompose
    int d = INPUT_DIMENSION - INPUT_RANK;
    MatrixXd p_all, r_all;
    MatrixXd diff_corner_val0, idx_corner, w_no_select, r, s, l, p, w_next, idx_gen, diff_corner_val1, idex_init_corner;
    MatrixXd idx_corner_part, diff_corner_val2, idx_init_no_corner, p_no_corner_part, pp_test, pp_test_asd_idx, pp_test_dsd_idx;
    MatrixXd idx_no_corner_choice, idx_init_no_corner, idx_corner_add_part, idx_corner, w_no_use, l;
    MatrixXd w_use = eignval_W;
    int count_num = 0;
    double p = 1.0;
    while(p > 1e-7) {
        count_num++;
        diff_corner_val0 = w_use - w_use.sum() / d;
        if (diff_corner_val0.all() <= -1e-6){
//            idx_cor



        }
    }
}


OnlinePCAReturn OnlinePCA(int INPUT_DIMENSION, int INPUT_RANK, double eta, double alpha, MatrixXd Lt, MatrixXd w_last, double AccumulatePCA, MatrixXd PLast) {
    OnlinePCAReturn PCA;
    JacobiSVD<Eigen::MatrixXf> svd(Lt, ComputeThinU | ComputeThinV );
    MatrixXd eignvec_W, eignval_W, eigvec_W_h;
    eignvec_W = svd.matrixU();
    eigvec_W_h = svd.matrixV();
    eignval_W = svd.singularValues();
    // J = U\SigmaV^T
    MatrixXd Identity;
    Identity.setIdentity(INPUT_DIMENSION, INPUT_DIMENSION);
    AccumulatePCA += ((Identity - PLast) * Lt * Lt.transpose()).trace();
    // mixture_decompose_lift
    MatrixXd r_candidate, p_dist;
    mixture_decompose_list(eignval_W, INPUT_DIMENSION, INPUT_RANK, r_candidate, p_dist);

    return PCA;
}


