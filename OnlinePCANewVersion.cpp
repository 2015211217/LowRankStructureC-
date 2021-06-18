//
// Created by holmes on 2021/5/31.
//

#include "OnlinePCANewVersion.h"
#include "math.h"
#include <unsupported/Eigen/MatrixFunctions>

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

MatrixXd BubbleSort(MatrixXd w_input) {
    MatrixXd rankWInput = w_input;
    for (int i = 0; i < rankWInput.cols() - 1;i++) {
        for (int j = i + 1; j < rankWInput.cols(); j++) {
            if (rankWInput(0, i) < rankWInput(0, j)) {
                double temp = rankWInput(0, j);
                rankWInput(0, j) = rankWInput(0, i);
                rankWInput(0, i) = temp;
            }
        }
    }
    return rankWInput;
}

double firstNorm(MatrixXd inputM) {
    double norm1 = 0.0;
    for (int i = 0;i < inputM.rows();i++)
        for (int j = 0;j < inputM.cols();j++)
            norm1 += abs(inputM(i, j));
    return norm1;
}

mdl_return mixture_decompose_list(MatrixXd w_Input, int INPUT_DIMENSION, int INPUT_RANK) {
//    w_input = w_input.transpose();
//    cout << "the output " << w_Input.rows() << w_Input.cols() << endl;
    //the out put 41
    int d = INPUT_DIMENSION - INPUT_RANK;
    MatrixXd r, p, currentR;
//    r.resize(0, INPUT_DIMENSION);
    MatrixXd w_input = w_Input;
//    p.resize(1, 0);
    while(firstNorm(w_input) > 1e-3) {
        currentR.resize(1, INPUT_DIMENSION);
        currentR.fill(0);
        double norm1WInput = firstNorm(w_input);
        int countR = 0;
        for(int i = 0;i < w_input.rows();i++) {
            if (w_input(i, 0) == norm1WInput / (1.0 * d)) {// we should add this col into the r
                currentR(0, i) = norm1WInput / (1.0 * d);
                countR++;
            }
        }
        // ranking the w_input
        // But we need to know the sequence of it.
        MatrixXd rankWInput = w_input;
        while (countR < d) {
            //
            double temp = 0.0;
            int notation;
            for(int i = 0;i < w_input.rows();i++) {
                if (rankWInput(i, 0) != norm1WInput / (1.0 * d) && rankWInput(i, 0) > temp) {
                    temp = rankWInput(i, 0);
                    notation = i;
                }
            }
            rankWInput(notation, 0) = 0;
            currentR(0, notation) = 1/(1.0 * d);
            countR++;
        }
        for (int i = 0 ;i < currentR.cols();i++) {
            if (currentR(0, i) > 0)
                currentR(0, i) = 1 / (1.0 * d);
        }
        // Two places to modify
        MatrixXd SortWeight = BubbleSort(w_input.transpose());
        //  s: smallest of chosen; l: the largest one for not chosen.
        double s, l, currentP;
        s = SortWeight(0, d - 1);
        l = SortWeight(0, d);
//        cout << SortWeight <<endl;
        // renew weight and adding p
        if (d * s < (norm1WInput - d * l)) currentP = d*s;
        else currentP = norm1WInput - d * l;
        for (int i = 0; i < w_input.rows(); i++)
            w_input(i, 0) -= currentP * currentR(0, i);
        p.conservativeResize(1, p.cols() + 1);
        p(0, p.cols() - 1) = currentP;
        r.conservativeResize(r.rows() + 1, INPUT_DIMENSION);
        r.row(r.rows()-1) = currentR;
//        cout << w_input << endl;

    }
    mdl_return mdl;
    mdl.r_candidate = r;
    mdl.p_dist = p;
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
    p_mediate.fill(0);
    for(int i = 0; i < p_dist.cols();i++)
        p_mediate(0, i) = p_dist(0, i) / (p_dist.sum() * 1.0);

    idx_pick = randomChoose(p_mediate, 1, p_dist.cols(), false);
    int whatever = idx_pick(0, 0);

    MatrixXd Mat_corner, Proj_use_mat, w_hat, w_hat_svd, wReal, PReal;
    MatrixXd r_corner_diag;
    r_corner_diag.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);
    Proj_use_mat.conservativeResize(INPUT_DIMENSION, INPUT_DIMENSION);

    for (int i = 0;i < INPUT_DIMENSION;i++) {
        for (int j = 0;j < INPUT_DIMENSION;j++)
            if (i == j) r_corner_diag(i, i) = r_candidate(whatever, i);
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

