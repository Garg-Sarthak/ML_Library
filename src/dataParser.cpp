#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "dataParser.h"

using namespace std;

void DataFrame::parseData(const string& path){
    string line;
    int lineCnt = 2;

    ifstream dataFile(path);
    try{
        if (!dataFile.is_open()){
            throw runtime_error("File not found");
        }

        getline(dataFile, line);
        stringstream ss(line); // Create a stringstream from the line
        string featureName;

        while (getline(ss, featureName, ',')) {
            feature_names.push_back(featureName);
        }
        
        num_features = feature_names.size();


        while (getline(dataFile,line)){
            int entryCnt = 1;
            vector<double> row;
            string currEntry;
            for (char c : line){
                if (c == ','){
                    try{
                        row.push_back(stod(currEntry));
                    }
                    catch(exception& e){
                        string error = "Invalid data in line : " + to_string(lineCnt);
                        throw runtime_error(error);
                    }
                    currEntry.clear();
                    entryCnt++;
                }else{
                    currEntry += c;
                }
            }
            try{
                row.push_back(stod(currEntry));
            }catch(exception& e){
                string error = "Invalid data in line : " + to_string(lineCnt);
    
                throw runtime_error(error);
            }
            if (entryCnt != num_features){
                // cerr<<"Missing/Incomplete Data in Input at line : "<<lineCnt<<endl;
                throw runtime_error("Data Missing");
            }
            dataFrame.push_back(row);
            currEntry.clear();
            lineCnt++;
        }
    } catch(exception& e){
        string err = "Missing Data at line : " + to_string(lineCnt);
        throw runtime_error(err);
    }
    dataFile.close();
}

void DataFrame::displayData() const{
    for (auto &row : dataFrame){
        for (auto &value : row){
            cout<< value << " ";
        }
        cout<<endl;
    }
}
