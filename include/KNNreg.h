#ifndef KNN_REG
#define KNN_REG

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "dataParser.h"
#include "dataUtils.h"

using namespace std;
using namespace Eigen;

class KNN {
public :
    bool normalization;

    KNN(bool normalization = false){
        this -> normalization = normalization;
    }

    VectorXd predict (string path, Matrix<double,Dynamic,Dynamic> X_test, int k){
        DataFrame df;
        df.parseData(path,true);
        BaseMatrix m(df.dataFrame,true);
        return predict(m.featureMatrix,m.targetMatrix,X_test,k);
    }

    VectorXd predict(Matrix<double,Dynamic,Dynamic> X, Matrix<double,Dynamic,1> Y, Matrix<double,Dynamic,Dynamic> X_test, int k){
        if (k <= 0 || k>X.rows()) throw runtime_error("Invalid value for k");

        if (X.rows() != Y.rows() || (X.cols() != X_test.cols())) {
            cout << X << endl;
            throw runtime_error("Dimension Mismatch");
        }

        VectorXd Y_test (X_test.rows());

        for (int i = 0; i<X_test.rows(); i++){
            priority_queue<pair<double,double> > pq;

            for (int j = 0; j<X.rows(); j++){
                double dist = (X.row(j) - X_test.row(i)).squaredNorm();
                pq.push({dist,Y(j)});
                if (pq.size() > k){
                    pq.pop();
                }

            }

            double ans = 0;
            while(!pq.empty()){
                ans += pq.top().second;
                pq.pop();
            }
            ans /= k;
            Y_test(i) = ans;
        }
        cout << Y_test.transpose() << endl;
        return Y_test;
    }


};

#endif