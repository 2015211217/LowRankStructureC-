//
// Created by holmes on 2021/3/29.
//
#include "OnlinePCA.h"
#include "math.h"
#include <unsupported/Eigen/MatrixFunctions>

//using namespace MatrixBase;

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
    array_index.conservativeResize(1, array_len);
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

MatrixXi randomChoose(MatrixXd probability, int size, int len, bool Replace) { //Choose some from some of that
    MatrixXi randomReturn;
    randomReturn.conservativeResize(1, size);
    // all the replace is false in this algorithm, so we just skip the TRUE side
    double temp = 0;
    for (int i = 0; i < probability.cols();i++) {
        temp += probability(0, i);
        probability(0, i) = temp;
    }
    int cols = probability.cols();
    probability.conservativeResize(1, cols + 1);

    for (int i = 0; i < probability.cols()-1;i++)
        probability(0, probability.cols() - i - 1) = probability(0, probability.cols() - i - 2);
    probability(0 ,0) = 0;
    int count = 0;
    while (count < size) {
       double randomI = rand() / double(RAND_MAX);
       for (int j = 0; j < probability.cols()-1; j++) {
           if (randomI >= probability(0, j) and randomI <= probability(0, j + 1)) {
               int k = 0;
               for (; k < count ;k++)
                   if (randomReturn(0, k) == j) break;
               if (k == count) {
                   randomReturn(0, count) = j;
                   count++;
               }
           }
       }
   }

    return randomReturn;
}

struct mdl_return {
    MatrixXd r_candidate;
    MatrixXd p_dist;
};

mdl_return mixture_decompose_list(MatrixXd w_input, int INPUT_DIMENSION, int INPUT_RANK) {
//    w_input = w_input.transpose();
    int d = INPUT_DIMENSION - INPUT_RANK;
    MatrixXd p_all;
    MatrixXd r_all;
    MatrixXd diff_corner_val0, w_no_select, r, w_next, idx_gen, diff_corner_val1, idex_init_corner;
    MatrixXd idx_corner_part, diff_corner_val2, idx_init_corner, p_no_corner_part, pp_test;
    MatrixXd idx_corner_add_part, w_no_use;
    MatrixXi idx_corner;
    MatrixXi idx_no_corner_choice, pp_test_asd_idx, pp_test_dsd_idx;
    MatrixXd idx_init_no_corner;
    MatrixXd w_use = w_input.transpose();

    double s, l ,p = 1;
    int count_num = 0;
    int countPR = 0;
    while(p > 1e-7) {
        count_num++;
        bool flagDiffVal0 = false;
        diff_corner_val0.conservativeResize(1, INPUT_DIMENSION);
        for (int i = 0;i < INPUT_DIMENSION;i++) {
            diff_corner_val0(0, i) = w_use(0, i) - w_use.sum() / (d * 1.0);
            if(diff_corner_val0(0, i) <= 0) flagDiffVal0 = true;
        }

        if (flagDiffVal0){
            MatrixXd PInside;
            PInside.conservativeResize(1, w_use.cols());
            for (int i = 0; i < w_use.cols();i++)
                PInside(0, i) = w_use(0, i) / (w_use.sum()* 1.0);
            idx_corner = randomChoose(PInside, d, w_use.rows(), false);
            w_no_select.conservativeResize(1, INPUT_DIMENSION);
            for (int i = 0; i < INPUT_DIMENSION; i++) {
                bool flag = true;
                for (int j = 0;j < idx_corner.cols();j++)
                    if (idx_corner(0, j) == i)
                        flag = false;
                if (!flag)
                    w_no_select(0, i) = 0;
                else
                    w_no_select(0, i) = abs(w_use(0, i));
            }
            r.conservativeResize(1, INPUT_DIMENSION);
//            cout << idx_corner << "idx_corner" <<endl;

            for (int i = 0;i < INPUT_DIMENSION;i++) {
                bool flagwhatsoever = false;
                for (int j = 0;j < idx_corner.cols();j++)
                    if(i == idx_corner(0, j)) flagwhatsoever = true;
                if(flagwhatsoever) r(0, i) = 1;
                else r(0, i) = 0;
            }

            MatrixXd w_use_idx_corner;
            w_use_idx_corner.conservativeResize(1, idx_corner.cols());
            w_use_idx_corner.fill(0);
            for (int i = 0;i < idx_corner.cols();i++)
                w_use_idx_corner(0, i) = w_use(0, idx_corner(0, i));
            s = w_use_idx_corner.minCoeff();
            l = w_no_select.maxCoeff();
            if (s > ((w_use.sum() / (d * 1.0)) - l)) p = ((w_use.sum() / (d* 1.0)) - l);
            else p = s;
            p_all.conservativeResize(1, countPR + 1);
            r_all.conservativeResize(countPR + 1, INPUT_DIMENSION);
            p_all(0, countPR) = p;
            r_all.row(countPR) = r;
            countPR++;
            for (int i = 0;i < w_use.cols();i++)
                w_use(0, i) = w_use(0, i) - p * r(0, i);
            int countQuit = 0;
            for (int i = 0;i < w_use.cols();i++)
                if (w_use(0, i) > 1e-7) countQuit ++;
            if (countQuit < d) break;

        } else {
            idx_gen.conservativeResize(1, INPUT_DIMENSION);
            diff_corner_val1.conservativeResize(1, INPUT_DIMENSION);
            for (int i = 0;i < INPUT_DIMENSION;i++) idx_gen(0, i) = i;
            MatrixXd tempSum;
            tempSum.conservativeResize(w_use.rows(), w_use.cols());
            tempSum.fill(w_use.sum() / (INPUT_RANK* 1.0));
            diff_corner_val1 = (w_use - tempSum);

            int temp = 0;
            idx_init_corner.conservativeResize(1, INPUT_DIMENSION);

            for (int i = 0;i < INPUT_DIMENSION;i++)
                if(diff_corner_val1(0, i) <= 1e-6) {
                    idx_init_corner(0, temp) = diff_corner_val1(0, i);
                    temp++;
                }

            idx_init_corner = sortMatrixXd(idx_init_corner);
            idx_corner_part.conservativeResize(1, idx_init_corner.cols());

            for (int i = 0;i < idx_init_corner.cols();i++) {
                idx_corner_part(0, i) = idx_gen(0, (int)idx_init_corner(0, i));
            }

            if(idx_corner_part.cols() < d) { // We need more
                int diff_num = d - idx_corner_part.rows();
                MatrixXd tempSum;
                tempSum.conservativeResize(w_use.rows(), w_use.cols());
                tempSum.fill(w_use.sum() / (d* 1.0));
                diff_corner_val2 = (w_use - tempSum).array().abs();
                int temp = 0;
                for (int i = 0; i < INPUT_DIMENSION; i++)
                    if (diff_corner_val2(0, i) > 1e-6) {
                        idx_init_no_corner(0, temp) = diff_corner_val2(0, i);
                        temp++;
                    }
                for (int i = 0; i < idx_init_no_corner.cols(); i++)
                    p_no_corner_part(0, temp) = w_use(0, idx_init_no_corner(0, i));
                pp_test = (p_no_corner_part).array().abs();
                pp_test_asd_idx = argsort(pp_test);
                pp_test_dsd_idx = pp_test_asd_idx;
                if (pp_test_dsd_idx(0, diff_num - 1) < 1e-8) break;
                MatrixXd p_idx_no_corner_choice;
                p_idx_no_corner_choice.conservativeResize(0, p_no_corner_part.cols());

                for (int i = 0; i < p_no_corner_part.cols();i++)
                    p_idx_no_corner_choice(0, i) = p_no_corner_part(0, i) / (p_no_corner_part.sum()* 1.0);
                idx_no_corner_choice = randomChoose(p_idx_no_corner_choice, diff_num, p_no_corner_part.cols(), false);

                for (int i = 0; i < idx_no_corner_choice.cols(); i++)
                    idx_corner_add_part(0, i) = idx_init_no_corner(0, idx_no_corner_choice(0, i));
//                idx_corner_add_part;
                w_no_use = p_no_corner_part;
                for (int i = 0; i < idx_no_corner_choice.cols(); i++)
                    w_no_use(0, idx_no_corner_choice(0, i)) = 0.0;
                l = w_no_use.maxCoeff();
            }
            else l = 0.0;
            r.conservativeResize(1, INPUT_DIMENSION);
            r.fill(0);
            for(int i = 0;i < idx_corner.cols();i++)
                r(0, idx_corner(0, i)) = 1;

            MatrixXd w_use_idx_corner;
            w_use_idx_corner.conservativeResize(idx_corner.cols(), w_use.cols());
            int tempidx = 0;
            for (int i = 0;i < idx_corner.cols();i++) {
                w_use_idx_corner.row(temp) = w_use.row(idx_corner(0, i));
                tempidx++;
            }
            s = w_use_idx_corner.minCoeff();
            if (s < w_use.sum() / (d* 1.0) - l) p = s;
            else p = w_use.sum() / (d* 1.0) - l;
            p_all.conservativeResize(1, countPR + 1);
            r_all.conservativeResize(countPR + 1, INPUT_DIMENSION);
            p_all(0, countPR) = p;
            r_all.row(countPR) = r;
            countPR++;
            w_next = w_use - p * r;
            w_use = w_next;
            int countQuit = 0;
            for (int i = 0;i < w_use.cols();i++)
                if (w_use(0, i) > 1e-7) countQuit ++;
            if (countQuit < d) break;
        }
        }
        if (count_num > 3 * INPUT_DIMENSION)
            cout << "too many steps for decomposition, something wrong" <<endl;

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
    w_lift.conservativeResize(n, n);
    bool flag = true;
    for (int i =0 ;i < n;i++)
        for (int j = 0;j < n;j++) {
            w_lift(i,j) = w(i, j) * d;
            if(w_lift(i, j) > (1.0 / (d_upper* 1.0) + 1e-7))
                flag = false;
        }
    if (flag){
        for (int i = 0; i < n; i++)
            for (int j = 0;j < n;j++)
                if (abs(w_lift(i, j)) < 1e-10) w_lift(i, j) = 1e-10;
        return w_lift;
    }
    else {
        idx_asd = argsort(w_lift.col(0));
        idx_dsd.conservativeResize(1, idx_asd.cols());
        for (int i = 0;i < idx_asd.cols();i++)
            idx_dsd(0, i) = idx_asd(0, i);

        w_lift_dsd.conservativeResize(idx_asd.cols(), n);
        for (int i = 0;i < idx_asd.cols();i++) {

            w_lift_dsd.row(i) = w_lift.row((int)idx_dsd(0, i));

        }

    }
    MatrixXd w_tilde;
    w_tilde = w_lift_dsd;
    int i = 1;
    while (w_tilde.maxCoeff() > (1.0 / (d_upper * 1.0) + 1e-7)) {
        w_tilde.conservativeResize(1, n);
        w_tilde.fill(0);
        for(int i = 0;i < n;i++) w_tilde(0, i) = 1.0 / (d_upper * 1.0);
        double sum_remain;
        sum_remain = (w_lift_dsd.row(i)).sum(); //第几行
        for(int j = i; j < n;j++)
            w_tilde(0, j) = (d - i) * w_lift_dsd(0, j) / (sum_remain * 1.0);
        i++;
    }
    MatrixXd w_return;
    w_return.conservativeResize(n, n);
    w_return.fill(0);
    for (int j = 0; j < idx_dsd.cols();j++)
        w_return(0, idx_dsd(0, j)) = w_tilde(0, idx_dsd(0, j));
    for (int i = 0; i < n; i++)
        if (abs(w_return(0, i)) < 1e-10) w_return(0, i) = 1e-10;
    return w_return;
}

OnlinePCAReturn OnlinePCA(int INPUT_DIMENSION, int INPUT_RANK, double eta, double alpha, MatrixXd Lt, MatrixXd w_last, double AccumulatePCA, MatrixXd PLast) {
    OnlinePCAReturn PCA;
    JacobiSVD<MatrixXd> svd(w_last, ComputeThinU | ComputeThinV);
    MatrixXd eignvec_W(w_last.rows(), w_last.cols());
    MatrixXd eignval_W(1, w_last.cols());
    MatrixXd eigvec_W_h(w_last.cols(), w_last.cols());
    eignvec_W = svd.matrixU();
    eigvec_W_h = svd.matrixV();
    eignval_W = svd.singularValues();
    MatrixXd Identity;
    Identity.setIdentity(INPUT_DIMENSION, INPUT_DIMENSION);
    AccumulatePCA += ((Identity - PLast) * Lt.transpose() * Lt).trace();
    MatrixXd r_candidate, p_dist;
    mdl_return mdl;
    mdl = mixture_decompose_list(eignval_W, INPUT_DIMENSION, INPUT_RANK);
    r_candidate = mdl.r_candidate;
    p_dist = mdl.p_dist;

    MatrixXi idx_pick;
    MatrixXi r_corner; // idx_pick choose one node.
    MatrixXd p_mediate;
    p_mediate.conservativeResize(1, p_dist.cols());
    for(int i = 0; i < p_dist.cols();i++)
        p_mediate(0, i) = p_dist(0, i) / (p_dist.sum() * 1.0);
    idx_pick = randomChoose(p_mediate, 1, p_dist.cols(), false);

    r_corner.conservativeResize(1, r_candidate.cols());
    for(int i  = 0;i < r_candidate.cols();i++) {
        r_corner(0, i) = r_candidate(idx_pick(0, 0), i);
    }

    if(abs(r_corner.sum() - (INPUT_DIMENSION - INPUT_RANK)) > 1e-6) cout << "wrong corner" <<endl;

    MatrixXd Mat_corner, Proj_use_mat, w_hat, w_hat_svd, wReal, PReal;
    MatrixXd r_corner_diag;
    r_corner_diag.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    Proj_use_mat.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    for (int i = 0;i < INPUT_DIMENSION;i++) {
        for (int j = 0;j < INPUT_DIMENSION;j++)
            if (i == j) r_corner_diag(i, i) = r_corner(0, i);
            else r_corner_diag(i, j) = 0;
    }
    Mat_corner = eignvec_W * r_corner_diag * eignvec_W.transpose();
    MatrixXd eye;
    eye.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    for (int i = 0;i < INPUT_DIMENSION;i++)
        for (int j = 0;j < INPUT_DIMENSION;j++)
            if(i == j) eye(i,j) = 1;
            else eye(i,j) = 0;
    for (int i = 0;i < INPUT_DIMENSION;i++)
        for(int j =0;j < INPUT_DIMENSION;j++)
            Proj_use_mat(i, j) = eye(i, j) - (INPUT_DIMENSION - INPUT_RANK) * Mat_corner(i ,j);
    MatrixXd w_last_log;
    w_last_log.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    w_last_log = w_last.log();
    MatrixXd eta_lt_lt;
    MatrixXd lt_outer;
    lt_outer.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    lt_outer = Lt.transpose() * Lt;
    eta_lt_lt.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    for (int i = 0;i < INPUT_DIMENSION;i++)
        for (int j = 0;j < INPUT_DIMENSION;j++)
            eta_lt_lt(i, j) = eta * lt_outer(i, j);
    w_hat.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    w_hat = w_last_log - eta_lt_lt;
    w_hat = w_hat.exp();
    w_hat_svd.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    for (int i = 0; i < INPUT_DIMENSION;i++)
        for (int j = 0;j < INPUT_DIMENSION;j++)
            w_hat_svd(i, j) = w_hat(i, j) / (w_hat.trace()* 1.0);
    w_last = capping_alg_lift(w_hat_svd, INPUT_DIMENSION, INPUT_RANK);
    PCA.PLast.conservativeResize(INPUT_DIMENSION,INPUT_DIMENSION);
    PCA.Preturn.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    PCA.W.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    PCA.PLast = Proj_use_mat;
    PCA.Preturn = Proj_use_mat;
    PCA.W = w_last;
    PCA.AccumulatePCA = AccumulatePCA;

    return PCA;
}

