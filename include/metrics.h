#ifndef METRICS_MODEL
#define METRICS_MODEL

#include <vector>
#include <Eigen/Dense>
#include <cmath>

using namespace Eigen;

class metrics {

    static double MAE(VectorXd Y_pred, VectorXd Y) {
        return ((Y_pred - Y).cwiseAbs().sum()) / (double)(Y.size());
    }

    static double MAE(VectorXi Y_pred, VectorXi Y) {
        return ((Y_pred - Y).cwiseAbs().sum()) / (double)(Y.size());
    }

    static double RMSE(VectorXd Y_pred, VectorXd Y) {
        return std::sqrt((Y_pred - Y).array().square().sum() / (double)Y.size());
    }

    static double RMSE(VectorXi Y_pred, VectorXi Y) {
        return std::sqrt((Y_pred - Y).array().square().sum() / (double)Y.size());
    }

    static double accuracy(VectorXi Y_pred, VectorXi Y) {
        return (Y_pred.array() == Y.array()).count() / (double)Y.size();
    }

    static double precision(VectorXi Y_pred, VectorXi Y) {
        int true_positives = ((Y_pred.array() == 1).select(Y.array() == 1, 0)).count();
        int predicted_positives = (Y_pred.array() == 1).count();
        return predicted_positives > 0 ? (double)true_positives / predicted_positives : 0.0;
    }

    static double recall(VectorXi Y_pred, VectorXi Y) {
        int true_positives = ((Y_pred.array() == 1).select(Y.array() == 1, 0)).count();
        int actual_positives = (Y.array() == 1).count();
        return actual_positives > 0 ? (double)true_positives / actual_positives : 0.0;
    }

    static double f1score(VectorXi Y_pred, VectorXi Y) {
        double p = precision(Y_pred, Y);
        double r = recall(Y_pred, Y);
        return (p + r > 0) ? 2 * p * r / (p + r) : 0.0;
    }

    static double precision_for_class(VectorXd Y_pred, VectorXd Y, int class_label) {
        int true_positives = ((Y_pred.array() == class_label).select(Y.array() == class_label, 0)).count();
        int predicted_positives = (Y_pred.array() == class_label).count();
        return predicted_positives > 0 ? (double)true_positives / predicted_positives : 0.0;
    }

    static double macro_precision(VectorXd Y_pred, VectorXd Y, int num_classes) {
        double sum_precision = 0.0;
        for (int i = 0; i < num_classes; ++i) {
            sum_precision += precision_for_class(Y_pred, Y, i);
        }
        return sum_precision / num_classes;
    }
};

#endif
