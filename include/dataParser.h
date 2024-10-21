#ifndef READ_DATA_H
#define READ_DATA_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>


// Class definition for DataFrame
class DataFrame {
private:
    std::vector<std::vector<double> > dataFrame; 

public:
    std::vector<std::string> feature_names;
    int num_features;
    void parseData(const std::string& path);       
    void displayData() const;                    
};

#endif 
