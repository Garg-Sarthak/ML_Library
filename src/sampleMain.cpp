#include <iostream>
#include "models.h"
#include "dataParser.h"

using namespace std;

int main(){
    DataFrame df;
    try{
        df.parseData("path_to_csv_file.csv",true);
    }catch(exception& e){
        cout<<"Error : "<<e.what()<<endl;
        return 0;
    }
    df.displayData();

    PCA pca;
    cout << pca.fit_transform("path_to_csv_file.csv",5) << endl;
}