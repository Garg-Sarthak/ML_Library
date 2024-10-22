#include "dataParser.h"
#include "makeMatrix.h"
#include <iostream>
#include <Eigen/Dense>



using namespace std;

int main(){
    bool isSupervised = true;
    DataFrame df;
    try{
        df.parseData("../Housing.csv",isSupervised);
    }catch(exception& e){
        cout<<"Error : "<<e.what()<<endl;
        return 0;
    }
    df.displayData();
    for (const auto& it : df.feature_names){
        cout << it<< endl;
    }

    BaseMatrix M(df.dataFrame,isSupervised);
    cout<<M.featureMatrix<<endl;
    cout<<M.targetMatrix<<endl;
    
}