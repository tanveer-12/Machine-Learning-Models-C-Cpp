/* ========== HEADER FILE
- DataPoint is a simple structure representing one row of our dataset.
- it contains a vector of features and a single target value.
- be defining it in a header, it can be used anywhere in the project

=========== */
#pragma once    //include this file only once no matter how many times its referenced
/*
could have used traditional guards
#ifndef DATA_POINT_H
#define DATA_POINT_H


#endif
*/
#include <vector>

struct DataPoint{
    std::vector<double> features;
    double target;
};
