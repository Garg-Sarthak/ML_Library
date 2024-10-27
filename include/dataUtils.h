#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <Eigen/Dense>

class DataUtils {
public:

    static Eigen::MatrixXd normalize(const Eigen::MatrixXd& matrix) {
        Eigen::MatrixXd normalized = matrix;
        for (int i = 0; i < matrix.cols(); ++i) {
            double mean = matrix.col(i).mean();
            // double std_dev = std::sqrt((matrix.col(i) - mean).squaredNorm() / matrix.rows());
            // normalized.col(i) = (matrix.col(i) - mean) / std_dev;
        }
        return normalized;
    }


};

#endif
