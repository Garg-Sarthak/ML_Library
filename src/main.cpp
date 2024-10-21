#include "dataParser.h"
#include <iostream>

using namespace std;

int main(){
    DataFrame df;
    try{
        df.parseData("../trial.csv");
    }catch(exception& e){
        cout<<"Error : "<<e.what()<<endl;
        return 0;
    }
    df.displayData();
    for (const auto& it : df.feature_names){
        cout << it<< endl;
    }
}