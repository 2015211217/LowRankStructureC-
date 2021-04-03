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
    MatrixXd diff_corner_val0, idx_corner, w_no_select, r, w_next, idx_gen, diff_corner_val1, idex_init_corner;
    MatrixXd idx_corner_part, diff_corner_val2, idx_init_no_corner, p_no_corner_part, pp_test, pp_test_asd_idx, pp_test_dsd_idx;
    MatrixXd idx_no_corner_choice, idx_corner_add_part, w_no_use, l;
    MatrixXd idx_corner;
    MatrixXd idx_init_no_corner;
    MatrixXd w_use = eignval_W;
    double s, l ,p;

    int count_num = 0;
    double p = 1.0;
    while(p > 1e-7) {
        count_num++;
        diff_corner_val0 = w_use - w_use.sum() / d;
        if (diff_corner_val0.all() <= -1e-6){
            default_random_engine generator;
            discrete_distribution<double> distribution {w_use / w_use.sum()};
//            for (int i = 0 ; i < w_use.size() ; i++) {
//                int number = distribution(generator);
//                ++
//            }
//  What the hell ?????????????
            for (int i = 0; i < d;i++) {

            }
            for (int i = 0; i < w_use.size(); i++)
                if (i in idx_corners) w_no_select(0, i) = 0;
                else w_no_select(0, i) = abs(w_use(0, i));
            r.resize(1, INPUT_DIMENSION);
            r.fill(0);
            for (int i = 0;i < INPUT_DIMENSION;i++)
                for (int j = 0;j < d;j++)
                    if(i == idx_corner[j]) r(0, i) = 1.0;
                    // 被选择的当中最小的

//            s = np.amin(w_use[idx_corner])
//            l = np.amax(w_no_select)
//            p = np.minimum(s, np.sum(w_use) / d - l)
//            p_all.append(p)
//            r_all.append(r)
//            w_next = w_use - p * r
//            w_use = w_next

        } else {

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

MatrixXd capping_alg_lift(MatrixXd w, int n, int k) {
    int d = n - k;
    double d_upper = 1.0;
    MatrixXd w_lift;
    w_lift = w * d;
    if (w_lift.all() < 1.0 / d_upper + 1e-7){
        for (int i = 0; i < w_lift.size(); i++)
            if (abs(w_lift(0, i)) < 1e-10) w_lift(0, i) = 1e-10;
        return w_lift;
    }
    else {

    }
}
