#ifndef KNN_CLS
#define KNN_CLS

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include "dataParser.h"
#include "dataUtils.h"

using namespace std;
using namespace Eigen;

class KNNClassifier{
public :
    bool normalization;

    KNNClassifier(bool normalization = false){
        this -> normalization = normalization;
    }

    VectorXi predict (string path, Matrix<double,Dynamic,Dynamic> X_test, int k){
        DataFrame df;
        df.parseData(path,true);
        BaseMatrix m(df.dataFrame,true);
        return predict(m.featureMatrix,m.targetMatrix,X_test,k);
    }

    VectorXi predict(Matrix<double,Dynamic,Dynamic> X, Matrix<double,Dynamic,1> Y, Matrix<double,Dynamic,Dynamic> X_test, int k){
        if (k <= 0 || k>X.rows()) throw runtime_error("Invalid value for k");

        if (X.rows() != Y.rows() || (X.cols() != X_test.cols())) {
            cout << X << endl;
            throw runtime_error("Dimension Mismatch");
        }

        if (normalization){
            DataUtils::normalize(X);
        }

        VectorXi Y_test (X_test.rows());

        for (int i = 0; i<X_test.rows(); i++){
            priority_queue<pair<double,double> > pq;

            for (int j = 0; j<X.rows(); j++){
                double dist = (X.row(j) - X_test.row(i)).squaredNorm();
                pq.push({dist,Y(j)});
                if (pq.size() > k){
                    pq.pop();
                }

            }

            unordered_map<int,int> mpp;
            while(!pq.empty()){
                int pred_label = pq.top().second;
                if (mpp.find(pred_label) == mpp.end()) mpp[pred_label] = 0;
                mpp[pred_label]++;
                pq.pop();
            }
            int ans = 0; int mxCnt = INT_MIN;
            for (auto it : mpp){
                if (it.second > mxCnt){
                    ans = it.first;
                    mxCnt = it.second;
                }
            }
            Y_test(i) = ans;
        }
        cout << Y_test.transpose() << endl;
        return Y_test;
    }


};

#endif