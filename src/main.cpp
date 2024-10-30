#include "dataParser.h"
#include "makeMatrix.h"
#include "linearRegression.h"
#include "logisticRegression.h"
#include "KNNreg.h"
#include "KNNclass.h"
#include <iostream>
#include <Eigen/Dense>



using namespace std;

int main(){
    bool isSupervised = true;
    DataFrame df;
    try{
        // df.parseData("../Salary_dataset.csv",isSupervised);
        df.parseData("../Social_Network_Ads.csv",isSupervised);
        
    }catch(exception& e){
        cout<<"Error : "<<e.what()<<endl;
        return 0;
    }

    BaseMatrix M(df.dataFrame,isSupervised);
    cout<<M.featureMatrix<<endl;

    // Eigen::Matrix2d h;
    // h << 1.5,6.3,2.4,4.4;
    // h.resize(1);
    // cout << h << endl;
    // cout << M.featureMatrix.head(15).transpose()<<endl;
    // cout<<M.featureMatrix.rows()<<endl;
    // cout<<M.featureMatrix.cols()<<endl;
    // LogisticRegression logReg = LogisticRegression(true,true);
    // logReg.fit("../Social_Network_Ads.csv",0.0001,100000);
    // KNN mod = KNN();
    // mod.predict("../Salary_dataset.csv",M.featureMatrix,3);

    KNNClassifier mod = KNNClassifier();
    mod.predict("../Social_Network_Ads.csv",M.featureMatrix,4);
    

}