#include "csv_loader.h"
#include <fstream>  // file input/output to read the CSV
#include <sstream>  // string stream to split CSV lines into columns
#include <iostream>


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
    // getline = reads one row/line at a time from the file into a string line
    while(std::getline(file, line)){        // gives me "5.1,3.5,1.4,0.2"
        // allows splitting a line by commas
        std::stringstream ss(line);
        // reads each value in the line
        std::string value;
        std::vector<double> features; // stores all columns except the target col
        double target = 0.0;
        int col = 0;
        // ss = std::stringstream(line)
        // reads one token at a time, separated by the delimiter ','
        // stores each token in the string value
        while(std::getline(ss, value, ',')){
            if(col == targetColumn){
                // stod = string to double
                target = std::stod(value);
            }
            else{
                features.push_back(std::stod(value));
            }
            col++;
        }
        // get the feature vec and target into a single DataPoint
        // adds it to the dataset vec
        // after the loop, dataset contains all rows of the csv as Datapoint(s)
        dataset.push_back({features, target});
    }
    return dataset;
}