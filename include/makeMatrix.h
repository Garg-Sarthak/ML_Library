#ifndef MAKE_MATRIX_H
#define MAKE_MATRIX_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

class BaseMatrix {
public:
    Matrix<double, Dynamic, Dynamic> featureMatrix;
    Matrix<double, Dynamic, 1> targetMatrix;
    int featureCnt;
    

    BaseMatrix(std::vector<std::vector<double>> dataFrame, bool isSupervised) {
        int rows = dataFrame.size();
        int cols = isSupervised ? dataFrame[0].size() - 1 : dataFrame[0].size();

        featureMatrix = Matrix<double, Dynamic, Dynamic>(rows, cols); 

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                featureMatrix(row, col) = dataFrame[row][col];
            }
        }

        if (isSupervised) {
            targetMatrix = Matrix<double, Dynamic, 1>(rows);
            for (int row = 0; row < rows; row++) {
                targetMatrix(row, 0) = dataFrame[row][cols - 1];
            }
        }
    }
};

#endif
