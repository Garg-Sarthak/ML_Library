#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "dataParser.h"

using namespace std;

class DataFrame{
    private : 
        vector<vector<double> > dataFrame;

    public :
        void parser(const string& path){
            string line;

            ifstream dataFile(path);
            try{
                if (!dataFile.is_open()){
                    throw runtime_error("File not found");
                }

                int lineCnt = 1;

                while (getline(dataFile,line)){
                    vector<double> row;
                    string currEntry;
                    for (int i = 0; i<line.size(); i++){
                        char c = line[i];
                        if (c == ','){
                            try{
                                row.push_back(stod(currEntry));
                            }
                            catch(exception& e){
                                string error = "Invalid data in line : " + to_string(lineCnt);
                                throw runtime_error(error);
                            }
                            currEntry.clear();
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
                    dataFrame.push_back(row);
                    currEntry.clear();
                    lineCnt++;
                }
            } catch(exception& e){
                cerr<< "Error while parsing data file : "<< e.what() << endl;
            }
            dataFile.close();
        }
    
        void displayData(){
            for (auto &row : dataFrame){
                for (auto &value : row){
                    cout<< value << " ";
                }
                cout<<endl;
            }
        }
};

int main(){
    
    return 0;

}
