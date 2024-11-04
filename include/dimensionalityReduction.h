#ifndef DIM_RED
#define DIM_RED

#include <Eigen/EigenValues>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>

#include "dataParser.h"
#include "makeMatrix.h"
#include "dataUtils.h"

using namespace Eigen;
using namespace std;


class PCA{
public :
    int k = 0;
    bool isSupervised = false;

    MatrixXd fit_transform(string path, int reduced_dims){
        DataFrame df;
        df.parseData(path,false);
        BaseMatrix m(df.dataFrame,false);
        return fit_transform(m.featureMatrix,reduced_dims);
    }

    MatrixXd fit_transform(MatrixXd dataMatrix,int reduced_dims){
        MatrixXd X = dataMatrix;

        int rows = X.rows();
        int cols = X.cols();

        if (reduced_dims > cols) reduced_dims = cols;

        //standardize
        for (int col = 0; col < cols; col++){
            double mean = X.col(col).mean();
            VectorXd A = (X.col(col) - VectorXd::Constant(rows,mean));
            double std_dev = sqrt((A.cwiseProduct(A)).sum() / (rows-1));
            if (std_dev < 1e-10){
                cerr << "Standard deviation close to zero";
                std_dev = 1e-10;
            }
            X.col(col) = A/std_dev;
        }

        MatrixXd covarMatrix;
        covarMatrix = (X.transpose()*X)/(rows-1);

        // compute eigenvalue
        SelfAdjointEigenSolver<MatrixXd> eigensolver(covarMatrix);
        VectorXd eigenvalues = eigensolver.eigenvalues();
        MatrixXd eigenvectors = eigensolver.eigenvectors();

        // pair eigen value  - eigen vector
        vector<pair<double, VectorXd>> eigenPairs;
        for (int i = 0; i < eigenvalues.size(); ++i) {
            eigenPairs.emplace_back(eigenvalues(i), eigenvectors.col(i));
        }
        // to get the top k eigenvectors
        sort(eigenPairs.begin(), eigenPairs.end(),
                [](const pair<double, VectorXd>& a, const pair<double, VectorXd>& b) {
                    return a.first > b.first;
                });

        MatrixXd topKEigenvectors(cols, reduced_dims);
        for (int i = 0; i < reduced_dims; ++i) {
            topKEigenvectors.col(i) = eigenPairs[i].second;
        }

        MatrixXd reducedData = X * topKEigenvectors;

        return reducedData;
    }
};

// class LDA{
//     bool isSupervised = true;
    
//     MatrixXd fit_transform()
// };

#endif