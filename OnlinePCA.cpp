//
// Created by holmes on 2021/3/29.
//
#include "OnlinePCA.h"

MatrixXd sortMatrixXd(MatrixXd a) {
    for (int i = 0;i < a.cols() - 1;i++)
        for (int j = i + 1;j < a.cols();j++) {
            if(a(0, i) > a(0, j)) {
                double temp = a(0, i);
                a(0, i) = a(0, j);
                a(0, j) = temp;
            }
        }
    return a;
}

MatrixXi argsort(MatrixXd array) {
    int array_len(array.cols());
    MatrixXi array_index;
//    MatrixXi array_index(array_len, 0);
    array_index.resize(1, array_len);
    for (int i = 0; i < array_len;i++)
        array_index(0, i) = i;

    for (int i = 0;i < array_len - 1;i++)
        for (int j = i + 1; j < array_len ; j++) {
            if (array(0, i) > array(0, j)) {
                double temp;
                int tempInt;
                temp = array(0, i);
                array(0, j) = array(0, i);
                array(0, i) = temp;
                tempInt = array_index(0, i);
                array_index(0, j) = array_index(0, i);
                array_index(0, i) = tempInt;
            }
        }
    return array_index;
}

MatrixXi randomChoose(MatrixXd probablility, int size, int len, bool Replace) { //Choose some from some of that
    MatrixXi randomReturn;
    // all the replace is false in this algorithm, so we just skip the TRUE side
    double temp = 0;
    for (int i = 0; i < probablility.cols();i++) {
        temp += probablility(0, i);
        probablility(0, i) = temp;
    }
    for (int i = 0; i < size;i++) {
        double randomI = rand() / double(RAND_MAX);
        for (int j = 0;j < probablility.cols() - 1;j++) {
            if (randomI <= probablility(0, j) and randomI >= probablility(0, j + 1)) {
                int k = 0;
                for (k = 0; k < randomReturn.cols(); k++)
                    if (randomReturn(0, k) == j + 1) break;
                if (k == randomReturn.cols()) {
                    randomReturn(0, k - 1) = j + 1;
                } else i--;
            }
        }
    }
    return randomReturn;
}

struct mdl_return {
    MatrixXd r_candidate;
    MatrixXd p_dist;
};

mdl_return mixture_decompose_list(MatrixXd eignval_W, int INPUT_DIMENSION, int INPUT_RANK) {
    int d = INPUT_DIMENSION - INPUT_RANK;
    MatrixXd p_all;
    MatrixXd r_all;
    MatrixXd diff_corner_val0, w_no_select, r, w_next, idx_gen, diff_corner_val1, idex_init_corner;
    MatrixXd idx_corner_part, diff_corner_val2, idx_init_corner, p_no_corner_part, pp_test;
    MatrixXd idx_corner_add_part, w_no_use;
    MatrixXi idx_corner;
    MatrixXi idx_no_corner_choice, pp_test_asd_idx, pp_test_dsd_idx;
    MatrixXd idx_init_no_corner;
    MatrixXd w_use = eignval_W;
    double s, l ,p;

    int count_num = 0;
    int countPR = 0;
    while(p > 1e-7) {
        count_num++;
        MatrixXd w_use_sum;
        w_use_sum.resize(1, w_use.cols());
        w_use_sum.fill(w_use.sum() / d);
        diff_corner_val0 = w_use - w_use_sum;
        if (diff_corner_val0.all() <= -1e-6){
            int w_use_len = eignval_W.cols();
            MatrixXd PInside = (MatrixXd)(w_use / w_use.sum());

            idx_corner = randomChoose(PInside, d, w_use.cols(), false);

            for (int i = 0; i < w_use.size(); i++) {
                bool flag = true;
                for (int j = 0;j < idx_corner.cols();j++)
                    if (idx_corner(0, j) == (i + 1)) flag = false;
                if (!flag) w_no_select(0, i) = 0;
                else w_no_select(0, i) = abs(w_use(0, i));
            }
            r.resize(1, INPUT_DIMENSION);
            r.fill(0);
            for (int i = 0;i < INPUT_DIMENSION;i++)
                for (int j = 0;j < d;j++)
                    if(i == idx_corner(0, j)) r(0, i) = 1.0;
            MatrixXd w_use_idx_corner;
            for (int i = 0;i < idx_corner.size();i++)
                w_use_idx_corner(0, i) = w_use(0, idx_corner(0, 1));
            s = w_use_idx_corner.minCoeff();
            l = w_no_select.maxCoeff();
            p = s > w_use.sum() / d - l ? s : w_use.sum() / d - l;
            p_all(0, countPR) = p;
            r_all.row(countPR) = r;

            countPR++;
            w_next = w_use - p * r;
            w_use = w_next;
        } else {
            MatrixXd idx_gen;
            for (int i = 1;i <= INPUT_DIMENSION;i++) idx_gen(0, i) = i;
            MatrixXd tempSum;
            tempSum.resize(1, w_use.cols());
            tempSum.fill(w_use.sum() / INPUT_RANK);

            diff_corner_val1 = (w_use - tempSum).cwiseAbs();
            //             idx_init_corner = list(np.where(diff_corner_val1 <= 1e-6)[0])
            int temp = 0;
            for (int i = 0;i < INPUT_DIMENSION;i++)
                if(diff_corner_val1(0, i) <= 1e-6) {
                    idx_init_corner(0, temp) = diff_corner_val1(0, i);
                    temp++;
                }
            temp = 0;
            idx_init_corner = sortMatrixXd(idx_init_corner);


            for (int i = 0;i < idx_init_corner.size();i++)
                idx_corner_part(0, temp) = idx_gen(0, idx_init_corner(0, i));
            if(idx_corner_part.size() < d) { // We need more
                int diff_num = d - idx_corner_part.size();
                MatrixXd tempSum;
                tempSum.resize(1, w_use.cols());
                tempSum.fill(w_use.sum() / d);
                diff_corner_val2 = (w_use - tempSum).cwiseAbs();
                int temp = 0;
                for (int i = 0; i < INPUT_DIMENSION; i++)
                    if (diff_corner_val2(0, i) > 1e-6) {
                        idx_init_no_corner(0, temp) = diff_corner_val2(0, i);
                        temp++;
                    }
                for (int i = 0; i < idx_init_no_corner.size(); i++)
                    p_no_corner_part(0, temp) = w_use(0, idx_init_no_corner(0, i));
                pp_test = (p_no_corner_part).cwiseAbs();
                pp_test_asd_idx = argsort(pp_test);
                pp_test_asd_idx = pp_test_asd_idx.reverse();
                if (pp_test_dsd_idx(0, diff_num - 1) < 1e-8) break;

                MatrixXd p_idx_no_corner_choice;
                p_idx_no_corner_choice.resize(0, p_no_corner_part.cols());
                for (int i = 0; i < p_no_corner_part.cols();i++)
                    p_idx_no_corner_choice(0, i) = p_no_corner_part(0, i) / p_no_corner_part.sum();
                idx_no_corner_choice = randomChoose(p_idx_no_corner_choice, diff_num, p_no_corner_part.cols(), false);

                for (int i = 0; i < idx_no_corner_choice.size(); i++)
                    idx_corner_add_part(0, i) = idx_init_no_corner(0, idx_no_corner_choice(0, i));
//                idx_corner_add_part;
                w_no_use = p_no_corner_part;
                for (int i = 0; i < idx_no_corner_choice.size(); i++)
                    w_no_use(0, idx_no_corner_choice(0, i)) = 0.0;
                l = w_no_use.maxCoeff();
            }
            else l = 0.0;
            r.resize(1, INPUT_DIMENSION);
            r.fill(0);
            for(int i = 0;i < idx_corner.size();i++)
                r(0, idx_corner(0, i)) = 1.0;

            s = w_use(0, idx_corner(0,0));
            for(int i = 0;i < idx_corner.size();i++)
                if(w_use(0, idx_corner(0,i)) < s) s = w_use(0, idx_corner(0,i));
            p = s < w_use.sum() / d - l ? s : w_use.sum() / d - l;
            p_all(0, countPR) = p;
            r_all.row(countPR) = r;
            countPR++;
            w_next = w_use - p * r;
            w_use = w_next;
        }
        }
        if (count_num > 3 * INPUT_DIMENSION) {
            cout << "too many steps for decomposition, something wrong" <<endl;
        }
    mdl_return mdl;
    mdl.r_candidate = r_all;
    mdl.p_dist = p_all;
    return mdl;
}

MatrixXd capping_alg_lift(MatrixXd w, int n, int k) {
    int d = n - k;
    double d_upper = 1.0;
    MatrixXd w_lift, w_lift_dsd;
    MatrixXi idx_asd, idx_dsd;

    w_lift = w * d;
    if (w_lift.all() < 1.0 / d_upper + 1e-7){
        for (int i = 0; i < w_lift.size(); i++)
            if (abs(w_lift(0, i)) < 1e-10) w_lift(0, i) = 1e-10;
        return w_lift;
    }
    else {
        idx_asd = argsort(w_lift);
        for (int i = 0;i < idx_asd.cols();i++)
            idx_dsd(0, i) = idx_asd(0, idx_asd.cols() - 1 - i);


        for (int i = 0;i < idx_dsd.size();i++) {
            w_lift_dsd(0, i) = w_lift(0, idx_dsd(0, i));
        }
    }
    MatrixXd w_tilde;
    w_tilde = w_lift_dsd;
    int i = 1;
    while (w_tilde.maxCoeff() > 1.0 / d_upper + 1e-7) {
        w_tilde.resize(1, n);
        w_tilde.fill(0);
        for(int i = 0;i < n;i++) w_tilde(0, i) = 1.0 / d_upper;
        double sum_remain;
        sum_remain = (w_lift_dsd.row(i)).sum(); //第几行
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
    JacobiSVD<MatrixXd> svd(w_last, ComputeThinU | ComputeThinV);
    MatrixXd eignvec_W(w_last.rows(), w_last.rows());
    MatrixXd eignval_W(1, w_last.cols());
    MatrixXd eigvec_W_h(w_last.cols(), w_last.cols());
    eignvec_W = svd.matrixU();
    eigvec_W_h = svd.matrixV();
    eignval_W = svd.singularValues();

    MatrixXd Identity;
    Identity.setIdentity(INPUT_DIMENSION, INPUT_DIMENSION);
    AccumulatePCA += ((Identity - PLast) * Lt * Lt.transpose()).trace();
    // mixture_decompose_lift
    MatrixXd r_candidate, p_dist;
    mdl_return mdl;

    mdl = mixture_decompose_list(eignval_W, INPUT_DIMENSION, INPUT_RANK);
    r_candidate = mdl.r_candidate;
    p_dist = mdl.p_dist;

    MatrixXi idx_pick;
    MatrixXi r_corner; // idx_pick choose one node.
    MatrixXd p_mediate;
    p_mediate.resize(0, p_dist.cols());
    for(int i = 0; i < p_dist.cols();i++)
        p_mediate(0, i) = p_dist(0, i) / p_dist.sum();
    idx_pick = randomChoose(p_mediate, 1, p_dist.cols(), false);

    int temp = 0;
    for(int i  = 0;i < idx_pick.size();i++){
        r_corner(0, temp) = r_candidate(0, idx_pick(0, i));
        temp++;
    }
    if(abs(r_corner.sum() - (INPUT_DIMENSION - INPUT_RANK)) > 1e-6) cout << "wrong corner" <<endl;
    MatrixXd Mat_corner, Proj_use_mat, w_hat, w_hat_svd, wReal, PReal;
    MatrixXd r_corner_diag;
    r_corner_diag.resize(1, r_corner.cols());
    for (int i = 0;i < r_corner.cols();i++) {
        r_corner_diag(0, i) = r_corner(i , i);
    }
    Mat_corner = eignvec_W * r_corner_diag * eignvec_W.transpose();
    MatrixXd eye;
    for (int i = 0;i < INPUT_DIMENSION;i++)
        for (int j = 0;j < INPUT_DIMENSION;j++)
            if(i == j) eye(i,j) = INPUT_DIMENSION;
            else eye(i,j) = 0;
    Proj_use_mat = eye - (INPUT_DIMENSION - INPUT_RANK) * Mat_corner;
    MatrixXd w_last_log;
    w_last_log.resize(1, w_last.cols());
    for(int i = 0;i < w_last.cols();i++)
        w_last_log(0, i) = log(w_last(0, i));
    MatrixXd eta_lt_lt;
    MatrixXd lt_outer;
    lt_outer = Lt * Lt.transpose();
    eta_lt_lt.resize(1, INPUT_DIMENSION);
    for (int i = 0;i < INPUT_DIMENSION;i++)
        for (int j = 0;j < INPUT_DIMENSION;j++)
            eta_lt_lt(i, j) = eta * lt_outer(i, j);
    w_hat = w_last_log - eta_lt_lt;
    for (int i = 0;i < w_hat.cols();i++)
        w_hat(0, i) = exp(w_hat(0, i));

    w_hat_svd = w_hat / w_hat.trace();
    w_last = capping_alg_lift(w_hat_svd, INPUT_DIMENSION, INPUT_RANK);
    PCA.PLast = Proj_use_mat;
    PCA.P = Proj_use_mat;
    PCA.W = w_last;
    PCA.AccumulatePCA = AccumulatePCA;
    return PCA;
}

