#include "dataParser.h"
#include "makeMatrix.h"
#include "linearRegression.h"
#include <iostream>
#include <Eigen/Dense>



using namespace std;

int main(){
    bool isSupervised = true;
    DataFrame df;
    try{
        df.parseData("../Salary_dataset.csv",isSupervised);
    }catch(exception& e){
        cout<<"Error : "<<e.what()<<endl;
        return 0;
    }
    // df.displayData();
    // for (const auto& it : df.feature_names){
    //     cout << it<< endl;
    // }

    BaseMatrix M(df.dataFrame,isSupervised);
    cout<<M.featureMatrix<<endl;
    cout<<M.featureMatrix.rows()<<endl;
    cout<<M.featureMatrix.cols()<<endl;
    LinearRegression model1 = LinearRegression("OLS",false);
    LinearRegression model = LinearRegression("Gradient_Descent",true);
    model.fit("../Salary_dataset.csv",0.001,1000,"Ridge");
    model1.fit("../Salary_dataset.csv");

    
}