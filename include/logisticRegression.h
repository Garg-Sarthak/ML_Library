#ifndef LOG_REG_H
#define LOG_REG_H

#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "dataParser.h"
#include "makeMatrix.h"
#include "dataUtils.h"

using namespace Eigen;

class LogisticRegression{
    public : 
        Matrix<double, Dynamic, 1> weights;
        double bias;
        int num_features;
        bool fit_intercept;
        bool normalization;

        LogisticRegression(bool fit_intercept=true,  bool normalization = false){
            bias = 0.0;
            num_features = -1;
            this->fit_intercept=fit_intercept;
            this->normalization = normalization;
        }

        void fit(std::string path,double learning_rate = 0.001,int max_iter=1000){
            DataFrame df;
            df.parseData(path,true);
            BaseMatrix m(df.dataFrame,true);
            fit(m.featureMatrix,m.targetMatrix,learning_rate,max_iter);
        }

        void fit(Matrix<double,Dynamic,Dynamic> X, Matrix<double,Dynamic,1> Y,double learning_rate = 0.001,int max_iter=1000){
            if (learning_rate < 0) throw std::runtime_error("Learning rate can't be negative");
            if (max_iter <= 0) {
                throw std::invalid_argument("max_iter must be a positive integer.");
            }
            if (X.rows() != Y.rows()){
                throw std::runtime_error("Dimension Mismatch: Feature matrix and target matrix must have same number of rows.");
            }
            if (normalization){
                X = DataUtils::normalize(X);
            }

            double dataPtCnt = X.rows(); 
            int featureCnt = X.cols();  
            weights = VectorXd::Zero(featureCnt); 
            bias = 0;

            for (int iter = 0; iter < max_iter; ++iter) {
                
                VectorXd linear_model = X * weights + VectorXd::Constant(dataPtCnt,bias);
                VectorXd predictions = sigmoid(linear_model);

                VectorXd dw = (1.0 / dataPtCnt) * (X.transpose() * (predictions - Y));
                double db = (1.0 / dataPtCnt) * (predictions - Y).sum();

                weights -= learning_rate * dw;
                if (fit_intercept) bias -= learning_rate * db;

            }
            std::cout << "weights : "<< weights.transpose() << std::endl;
            if (fit_intercept) std::cout<< "bias : " <<  bias << std::endl;
        }

        VectorXd predict_probabilities(const MatrixXd& X_test) {
            if (weights.size()==0 || weights.rows() != X_test.cols()){
                throw std::runtime_error("Predicting for untrained model or incorrect size for input matrix");
            }

            VectorXd linear_model = X_test * weights + VectorXd::Constant(X_test.rows(),bias) ;
            std :: cout << linear_model.transpose() << std::endl;
            VectorXd predictions = sigmoid(linear_model);
            return (predictions);
            
        }

private : 
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
        return 1.0 / (1.0 + (-z.array()).exp());
    }
};

#endif