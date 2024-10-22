#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

#include "dataParser.h"

using namespace std;

void DataFrame::parseData(const string& path, bool isSupervised){
    string line;
    int lineCnt = 2;

    ifstream dataFile(path);
    try{
        if (!dataFile.is_open()){
            throw runtime_error("File not found/Invalid Path Specified");
        }

        getline(dataFile, line);
        stringstream ss(line); 
        string label;

        while (getline(ss, label, ',')) {
            label_names.push_back(label);
            feature_names.push_back(label);
        }
        
        if (isSupervised){
            target_name = feature_names.back();
            feature_names.pop_back();
        }

        num_labels = label_names.size();
        


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
                        string error = "Invalid (NaN) data in line : " + to_string(lineCnt);
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
                string error = "Invalid (NaN) data in line : " + to_string(lineCnt);
    
                throw runtime_error(error);
            }
            if (entryCnt != num_labels){
                throw runtime_error("Data Missing (All rows must have same number of data points) ");
            }
            dataFrame.push_back(row);
            currEntry.clear();
            lineCnt++;
        }
    } catch(exception& e){
        string err = "Missing or NaN Data at line : " + to_string(lineCnt);
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
