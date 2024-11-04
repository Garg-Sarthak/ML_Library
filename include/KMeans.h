#ifndef KMCluster
#define KMCluster

#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <climits>
#include <vector>
#include <set>
#include <map>

#include "dataParser.h"
#include "makeMatrix.h"
#include "dataUtils.h"

using namespace std;
using namespace Eigen;


class KMeans{

public : 
    bool isSupervised = false;
    int K = 0;
    VectorXi clusterAssignments;
    MatrixXd centroids;         
    

    void fit(MatrixXd X, int K){
        if (K > X.rows()) K = X.rows();

        MatrixXd scaledX = DataUtils::normalize(X); // scaling/normalizing

        set<int> selected_indices; // initialse random centres 
        while (selected_indices.size() < K) {
            int index = rand() % scaledX.rows();
            selected_indices.insert(index); 
        }
        centroids = MatrixXd(K, X.cols());
        int i = 0;
        for (int index : selected_indices) {
            centroids.row(i++) = scaledX.row(index);
        }

        clusterAssignments = VectorXi::Constant(scaledX.rows(),-1);
        bool hasChanged = true; // convergence tracker

        while (hasChanged) {
            hasChanged = false;

            map<int, vector<int> > assignments;
            for (int i = 0; i < scaledX.rows(); i++) {
                double min_dist = numeric_limits<double>::max();
                int closestCentroid = -1;

                for (int j = 0; j < K; j++) {
                    VectorXd absDis = scaledX.row(i) - centroids.row(j);
                    double distance = absDis.squaredNorm();
                    if (distance < min_dist) {
                        min_dist = distance;
                        closestCentroid = j;
                    }
                }
                assignments[closestCentroid].push_back(i);
                if (clusterAssignments[i] != closestCentroid) {
                    clusterAssignments[i] = closestCentroid;
                    hasChanged = true; // Change detected
                }
            }

            // Update centroids
            for (int j = 0; j < K; j++) {
                if (!assignments[j].empty()) {
                    VectorXd newCentroid = VectorXd::Zero(X.cols());
                    for (int idx : assignments[j]) {
                        newCentroid += scaledX.row(idx).transpose();
                    }
                    centroids.row(j) = newCentroid / assignments[j].size();
                }
            }
        }
        isTrained = true;
    }

    vector<int> predict(MatrixXd X){
        if (!isTrained){
            throw runtime_error("Please train model before prediction");
        }
        MatrixXd scaledX = DataUtils::normalize(X);
        vector<int> predictedAssignments(scaledX.rows());

        for (int i = 0; i < scaledX.rows(); i++) {
            double min_dist = numeric_limits<double>::max();
            int closestCentroid = -1;

            for (int j = 0; j < centroids.rows(); j++) {
                VectorXd absDis = scaledX.row(i) - centroids.row(j);
                double distance = absDis.squaredNorm(); 
                
                if (distance < min_dist) {
                    min_dist = distance;
                    closestCentroid = j;
                }
            }

            predictedAssignments[i] = closestCentroid;
        }

        return predictedAssignments;

    }
private: 
    bool isTrained = false;
};

#endif