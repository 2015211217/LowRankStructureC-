//
// Created by holmes on 2021/3/29.
//

#include "OnlinePCA.h"
#include "Eigen/SVD"
#include <random>
#include "algorithm"
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

void mixture_decompose_list(MatrixXd eignval_W, int INPUT_DIMENSION, int INPUT_RANK, MatrixXd &r_candidate, MatrixXd &p_dist) {
    // start decompose
    int d = INPUT_DIMENSION - INPUT_RANK;
    MatrixXd p_all, r_all;
    MatrixXd diff_corner_val0, idx_corner, w_no_select, r, w_next, idx_gen, diff_corner_val1, idex_init_corner;
    MatrixXd idx_corner_part, diff_corner_val2, idx_init_corner, p_no_corner_part, pp_test, pp_test_asd_idx, pp_test_dsd_idx;
    MatrixXd idx_no_corner_choice, idx_corner_add_part, w_no_use, l;
    MatrixXd idx_corner;
    MatrixXd idx_init_no_corner;
    MatrixXd w_use = eignval_W;
    double s, l ,p;

    int count_num = 0;
    double p = 1.0;
    int countPR = 0;
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
            MatrixXd w_use_idx_corner;
            for (int i = 0;i < idx_corner.size();i++)
                w_use_idx_corner(0, i) = w_use(0, idx_corner(0, 1));
            s = w_use_idx_corner.min();
            l = w_no_select.max();
            p = s > w_use.sum() / d - l ? s : w_use.sum() / d - l;
            p_all(0, countPR) = p;
            r_all(0, countPR) = r;
            countPR++;
            w_next = w_use - p * r;
            w_use = w_next;
        } else {
            MatrixXd idx_gen;
            for (int i = 1;i <= INPUT_DIMENSION;i++) idx_gen(0, i) = i;
            diff_corner_val1 = abs(w_use - (w_use.sum()) / INPUT_RANK);
            //             idx_init_corner = list(np.where(diff_corner_val1 <= 1e-6)[0])
            int temp = 0;
            for (int i = 0;i < INPUT_DIMENSION;i++)
                if(diff_corner_val1(0, i) <= 1e-6) {
                    idx_init_corner(0, temp) = diff_corner_val1(0, i);
                    temp++;
                }
            temp = 0;
            idx_init_corner.sort();
            for (int i = 0;i < idx_init_corner.size();i++)
                idx_corner_part(0, temp) = idx_gen(0, idx_init_corner(0, i));
            if(idx_corner_part.size() < d) { // We need more
                int diff_num = d - idx_corner_part.size();
                diff_corner_val2 = abs(w_use - (w_use.sum() / d));
                int temp = 0
                for (int i = 0;i < INPUT_DIMENSION;i++)
                    if(diff_corner_val2(0, i) > 1e-6) {
                        idx_init_no_corner(0, temp) = diff_corner_val2(0, i);
                        temp++;
                    }
                for (int i = 0;i < idx_init_no_corner.size();i++)
                    p_no_corner_part(0, temp) = w_use(0, idx_init_no_corner(0, i));
                pp_test = abs(p_no_corner_part);
                pp_test_asd_jdx = pp_test_asd_idx.revserse();
                if (pp_test_dsd_idx(0, diff_num - 1) < 1e-8) break;

                // Another Random choice
                //                idx_no_corner_choice = rnd.choice(len(p_no_corner_part), size=diff_num,
                //                                                  replace=False, p=p_no_corner_part / np.sum(p_no_corner_part))
                for (int i = 0;i < idx_no_corner_choice.size();i++)
                    idx_corner_add_part(0, i) = idx_init_no_corner(0, idx_no_corner_choice(0, i));
                idx_corner = idx_corner_part   idx_corner_add_part;
                w_no_use = p_no_corner_part;
                for (int i = 0;i < idx_no_corner_choice.size();i++)
                    w_no_use(0, idx_no_corner_choice(0, i)) = 0.0;
                l = w_no_use.max;
            else {
                    idx_corner = idx_corner_part;
                    l = 0.0;
                }
            r.resize(1, INPUT_DIMENSION);
            r.fill(0);
            for(int i = 0;i < idx_corner.size();i++)
                r(0, idx_corner(0, i)) = 1.0;

            s = w_use(0, idx_corner(0,0));
            for(int i = 0;i < idx_corner.size();i++)
                if(w_use(0, idx_corner(0,i)) < s) s = w_use(0, idx_corner(0,i));
            p = s < w_use.sum() / d - l ? s : w_use.sum() / d - l;
            p_all(0, countPR) = p;
            r_all(0, countPR) = r;
            countPR++;
            w_next = w_use - p * r;
            w_use = w_next;
        }

        }
        if (count_num > 3 * INPUT_DIMENSION) {
            cout << "too many steps for decomposition, something wrong" <<endl;
            break;
        }
    }
    r_candidate = r_all;
    p_dist = p_all;
}

vector<int> argsort(MatrixXd array)
{
    int array_len(array.size());
    vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;
    sort(array_index.begin(), array_index.end(),
              [&array](int pos1, int pos2) {return (array(0, pos1) < array(0, pos2)); });
    return array_index;
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
        vector<int> idx_asd, idx_dsd;
        idx_asd = argsort(w_lift);
        idx_dsd = idx_asd.reverse();
        MatrixXd w_list_dsd;
        for (int i = 0;i < idx_dsd.size();i++) {
            w_lift_dsd(0, i) = w_lift(0, idx_dsd(0, i));
        }
    }
    MatrixXd w_tilde;
    w_tilde = w_lift_dsd;
    int i = 1;
    while (w_tilde.max() > 1.0 / d_upper + 1e-7) {
        w_tilde.resize(1, n);
        w_tilde.fill(0);
        for(int i = 0;i < n;i++) w_tilde(0, i) = 1.0 / d_upper;
        double sum_remain;
        sum_remain = (w_lift_dsd.col(i)).sum(); //第几行
        for(int j = i; j < n;j++)
            w_tilde(0, j) = (d - i) * w_lift_dsd(0, j) / sum_remain;
        i++;
    }
    MatrixXd w_return;
    w_return.resize(1, n);
    w_return.fill(0);
    for (int j = 0; j < idx_dsd.size();j++)
        w_return(0, idx_dsd(0, j)) = w_tilde(0, idx_dsd(0, j));
    for (int i = 0; i < n; i++)
        if (abs(w_return(0, i)) < 1e-10) w_return(0, i) = 1e-10;
    return w_return;
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

    // random choice
    // idx_pick = rnd.choice(len(p_dist), p = p_mediate)
    MatrixXd idx_pick, r_corner;
    int temp = 0;
    for(int i  = 0;i < idx_pick.size();i++){
        r_corner(0, temp) = r_candidate(0, idx_pick(0, i));
        temp++;
    }
    if(abs(r_corner.sum() - (INPUT_DIMENSION - INPUT_RANK)) > 1e-6) cout << "wrong corner" <<endl;
    MatrixXd Mat_corner, Porj_use_mat, w_hat, w_hat_svd, w_last, wReal, PReal;
    Mat_corner = eignvec_W * r_corner.diag() * eignvec_W.transpose();
    MatrixXd eye;
    for (int i = 0;i < INPUT_DIMENSION;i++)
        for (int j = 0;j < INPUT_DIMENSION;j++)
            if(i == j) eye(i,j) = INPUT_DIMENSION;
            else eye(i,j) = 0
    Proj_use_mat = eye - (INPUT_DIMENSION - INPUT_RANK) * Mat_corner;
    w_hat = exp(log(w_last) - eta * Lt.outer(Lt));
    w_hat_svd = w_hat / w_hat.trace();
    w_last = capping_alg_lift(w_hat_svd, INPUT_DIMENSION, INPUT_RANK);
    PCA.PLast = Proj_use_mat;
    PCA.P = Proj_use_mat;
    PCA.W = w_last;
    PCA.AccumulatePCA = AccumulatePCA;
    return PCA;
}

