#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <Eigen/Dense>

class DataUtils {
public:

    static Eigen::MatrixXd normalize(const Eigen::MatrixXd& matrix) {
        Eigen::MatrixXd X = matrix;
        int rows = X.rows();
        int cols = X.cols();
        for (int col = 0; col < cols; col++){
            double mean = X.col(col).mean();
            VectorXd A = (X.col(col) - VectorXd::Constant(rows,mean));
            double std_dev = sqrt((A.cwiseProduct(A)).sum() / (rows-1));
            if (std_dev < 1e-10){
                std::cerr << "Standard deviation close to zero";
                std_dev = 1e-10;
            }
            X.col(col) = A/std_dev;
        }
        return X;
    }


};

#endif
