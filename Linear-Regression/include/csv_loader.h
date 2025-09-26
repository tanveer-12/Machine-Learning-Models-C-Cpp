/* =========
csv_loader.h declares the loadCSV function, which reads a CSV
dataset and converts it into a vector of structured DataPoint(s).
By keeping this declaration in a header,multiple .cpp files can use
the loader without exposing the internal parsing logic
*/
#pragma once
#include <string>
#include <vector>
#include "data_point.h"

//reads a csv and converts it into a vector of DataPoint objects
// each row can have multiple features
// std::vector<DataPoint> -> returns a vector of Datapoint objects
// Each Datapoint represents one row in our dataset
// - feature -> vector of numeric feature values (double)
// - target -> single numeric target value (what we want to predict)

std::vector<DataPoint> loadCSV(const std::string &filename, int numFeatures, int targetColumn);

/* CODE BREAKDOWN FOR EASY UNDERSTANDING
1) const std::string &filename
- path to the csv file
- const -> so that the func does not modify the string
- & -> pass by reference to avoid copying the string (efficient than copying).

2) int numFeatures
- num of columns in the csv that we want as features
- lets the function know how many numbers per row should go into the features vector

3) int targetColumn
- which column in the csv should be the target value
- useful because our target might not always be the last column

5.1,3.5,1.4,0.2,Setosa

so my function would be like loadCSV("iris.csv", 4, 2)
feature - [5.1, 3.5, 0.2] (all cols except 2)
target - 1.4 (col 2)
this row becomes one DataPoint, added to the vector
*/