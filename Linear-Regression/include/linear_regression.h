#pragma once
#include <vector>
#include "data_point.h"

class LinearRegression{
private: 
    std::vector<double> weights;
    double bias;
public:
    // constructor to initialize values to weights and bias
    LinearRegression(int numFeatures);
    double predict(const std::vector<double> &features) const;
    void fit(const std::vector<DataPoint> &dataset, double lr, int epochs);
};

/* ==== CODE EXPLANATION
1) predict func:
- returns the yhat for a given feature vector
- pass by reference to avoid copying the vector
- const => so that func does not modify the features vec
- const at the end: so that the predict() func does not modify the LinearRegression obj (weights and bias)

2) fit func

- trains the model using gradient descent
- const std::vector<DataPoint> &dataset = all data points, passed by ref to avoid copying
- const so that fit func does not modify the dataset
- lr = learning rate, controls the step size
- epochs = number of iterations for training

*/