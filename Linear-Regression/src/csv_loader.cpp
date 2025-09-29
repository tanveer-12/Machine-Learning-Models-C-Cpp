#include "csv_loader.h"
#include <fstream>  // file input/output to read the CSV
#include <sstream>  // string stream to split CSV lines into columns
#include <iostream>
#include <map>
#include <algorithm>
#include <cctype>

// helper func to trim the whitespaces
std::string trim(const std::string &str){
    size_t first = str.find_first_not_of(" \t\r\n");
    if(first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}


// returns a vector of datapoint objects 
// takes filename, num of features, and target columns as input
std::vector<DataPoint> loadCSV(const std::string &filename, int numFeatures, int targetColumn){
    // prepare the dataset vector and open file
    std::vector<DataPoint> dataset;
    std::ifstream file(filename);   // open csv file
    if(!file.is_open()){
        std::cerr<<"Error opening file: "<< filename << "\n";
        return dataset;     // return empty dataset if file cannot be opened
    }

    // gonna read the  csv line by line
    std::string line;
    bool headerSkipped = true;   // to skip header row

    // map string labels to ints
    std::map<std::string, int> labelMap = {
        {"Setosa", 0},
        {"Versicolor", 1},
        {"Virginica", 2}
    };
    // getline = reads one row/line at a time from the file into a string line
    while(std::getline(file, line)){        // gives me "5.1,3.5,1.4,0.2"
        // allows splitting a line by commas
        if(!headerSkipped){
            headerSkipped = true;
            continue;
        }

        if(line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> tokens;

        // reads each value in the line
        // ss = std::stringstream(line)
        // reads one token at a time, separated by the delimiter ','
        // stores each token in the string value
        while(std::getline(ss, cell, ',')){
            cell = trim(cell);
            //remove surrounding quotes if present
            if(!cell.empty() && cell.front() == '"') cell.erase(0,1);
            if(!cell.empty() && cell.back() == '"') cell.pop_back();
            tokens.push_back(cell);
        }
        // skip rows with insufficient columns
        if((int)tokens.size() <= targetColumn) continue;

        DataPoint dp;
        // parse numeric features
        bool validRow = true;
        for(int i = 0; i < numFeatures; ++i){
            try {
                dp.features.push_back(std::stod(tokens[i]));
            } catch(...){
                validRow = false;
                break;  // skip this row if conversion fails
            }
        }
        if(!validRow) continue;

        // parse target column
        std::string labelStr = tokens[targetColumn];
        if(labelMap.find(labelStr) == labelMap.end()) continue;  // skip unknown labels
        dp.target = labelMap[labelStr];

        dataset.push_back(dp);
    }    
    return dataset;
}

// scale features to [0,1]
void scaleFeatures(std::vector<DataPoint>& dataset){
    if(dataset.empty()) return;
    int numFeatures = dataset[0].features.size();
    std::vector<double> minVals(numFeatures, 1e9);
    std::vector<double> maxVals(numFeatures, -1e9);

    //find min and max for each feature
    for(const auto& dp: dataset){
        for(int i=0; i<numFeatures; ++i){
            if(dp.features[i] < minVals[i])
            {
                minVals[i] = dp.features[i];
            }
            if(dp.features[i] > maxVals[i]){
                maxVals[i] = dp.features[i];
            }
        }
    }

    //scale
    for(auto& dp: dataset){
        for(int i=0; i<numFeatures; ++i){
            if(maxVals[i] != minVals[i]){
                dp.features[i] = (dp.features[i]-minVals[i]) / (maxVals[i]-minVals[i]);
            }
            else{
                dp.features[i] =0.0;
            }
        }
    }
}