#ifndef LIN_REG_H
#define LIN_REG_H

#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include "dataParser.h"
#include "makeMatrix.h"
#include "dataUtils.h"

using namespace Eigen;

class LinearRegression{
public : 
    Matrix<double, Dynamic, 1> weights;
    double bias;
    int num_features;
    bool fit_intercept;
    std::string regularization;
    bool normalization;
    std::string solver;
    int max_iter;
    float tolerance;

    LinearRegression(std::string solver = "OLS", std::string regularization = "None", int max_iter = 1000, bool fit_intercept=true,  bool normalization = false,  float tolerance = 1e-4){
        bias = 0.0;
        num_features = -1;
        if (regularization != "None" && regularization!= "Lasso" && regularization != "Ridge" && regularization != "Elastic_Net") {
            throw std::invalid_argument("Invalid regularization type. Must be 'None', 'Lasso', 'Ridge', or 'Elastic_Net'.");
        }
        if (max_iter <= 0) {
            throw std::invalid_argument("max_iter must be a positive integer.");
        }
        if (tolerance < 0) {
            throw std::invalid_argument("tolerance must be a non-negative float.");
        }
        if (solver != "Gradient_Descent" && solver != "OLS") {
            throw std::invalid_argument("Invalid solver. Must be 'Gradient_Descent' or 'OLS'.");
        }
        this->fit_intercept=fit_intercept;
        this->regularization = regularization;
        this->normalization = normalization;
        this->solver = solver;
        this->max_iter = max_iter;
        this->tolerance = tolerance;

        
    }

    void fit(std::string path,double learning_rate=0.001,double lambda = 0.1){
        DataFrame df;
        df.parseData(path,true);
        BaseMatrix m(df.dataFrame,true);
        fit(m.featureMatrix,m.targetMatrix,learning_rate,lambda);
    }

    void fit(Matrix<double,Dynamic,Dynamic> X, Matrix<double,Dynamic,1> Y,double learning_rate,double lambda = 0.1){
        if (X.rows() != Y.rows()){
            throw std::runtime_error("Dimension Mismatch: Feature matrix and target matrix must have same number of rows.");
        }

        if (normalization){
            X = DataUtils::normalize(X);
            Y = DataUtils::normalize(Y);
        }
        
        if (solver == "OLS"){
            if (fit_intercept) {
            Eigen::VectorXd b = Eigen::VectorXd::Ones(X.rows());
            X.conservativeResize(Eigen::NoChange, X.cols() + 1);
            X.col(X.cols() - 1) = b; 
            }
            OLS(X,Y);
        }
        else if (solver == "Gradient_Descent"){
            GD(X,Y,learning_rate,lambda);
        }
    }

private :
    void OLS(Matrix<double,Dynamic,Dynamic> X, Matrix<double,Dynamic,1> Y){

        Eigen::Matrix <double, Dynamic, 1> computedWeights;


        if ((X.transpose() * X).determinant() == 0) {
            throw std::runtime_error("Singular Data Matrix, can't be solved using ordinary least squares");
        }   


        computedWeights = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
        if (fit_intercept){
            weights = computedWeights.head(X.cols()-1);
            bias = computedWeights(X.cols()-1);
        }else{
            weights = computedWeights;
        }
        std :: cout << "Weights : " << weights.reshaped(1,weights.rows())<<std::endl;
        if (fit_intercept) std :: cout << "Bias : "  << bias << std :: endl;

    }

    void GD(Matrix<double,Dynamic,Dynamic> X, Matrix<double,Dynamic,1> Y, double learning_rate=0.001, double lambda=0.1){
        int featureCnt = X.cols();
        double dataPtCnt = (double) X.rows();


        weights = Eigen::VectorXd::Constant(featureCnt,0);


        Eigen::VectorXd predictedY;

        VectorXd penalty = Eigen::VectorXd::Constant(featureCnt,0);

        while (max_iter--){

            predictedY = X*weights + Eigen::VectorXd::Constant(dataPtCnt,bias);
            
            if (regularization == "Lasso"){
                penalty = lambda * (weights.cwiseAbs());
            }else if (regularization == "Ridge"){
                penalty = 2*lambda * weights;
            }else if (regularization == "Elastic_Net"){
                penalty = 2*lambda * weights + lambda * (weights.cwiseAbs());
            }

            weights = weights + (learning_rate * (X.transpose() * (Y - predictedY)) / dataPtCnt);
            if (fit_intercept){
                bias = bias + (learning_rate * ((Y - predictedY ).sum()) / dataPtCnt);
            }
        }

        std::cout << "weights : " << weights.reshaped(1,weights.rows())<<std::endl;
        std::cout<< "bias : " << bias<<std::endl;
    }

};

#endif