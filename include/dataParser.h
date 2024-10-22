#ifndef READ_DATA_H
#define READ_DATA_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>



class DataFrame {
public:
    std::vector<std::vector<double> > dataFrame; 
    std::vector<std::string> label_names;
    std::vector<std::string> feature_names;
    std::string target_name;
    int num_labels;
    void parseData(const std::string& path, bool isSupervised);    
    void displayData() const;                    
};

#endif 
