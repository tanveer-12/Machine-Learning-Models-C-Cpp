#include "linear_regression.h"  // includes the class definition
#include <cmath>
#include <iostream>


/*===========
constructor sets up the model with the correct number of weights and initializes all
parameters to zero
=============
*/
LinearRegression::LinearRegression(int numFeatures){
    weights.resize(numFeatures, 0.0);   // creates a vector of size numFeatures with weight=0.0
    bias = 0.0;     // sets the intercept to 0
}

/*===========
predict() calculates the model's output for a given input. it multiplies each feature by 
its corresponding weight, sums them, and then adds bias term.

Returns the predicted yhat(y^)
==============
weights = [0.5, -0.2], bias = 1.0
features = [2.0, 3.0]
yhat (predicted) = 1.0 + (0.5 * 2.0) + (-0.2 * 3.0) = 1.4
*/
double LinearRegression::predict(const std::vector<double>& features) const {
    double result = bias;
    // computing the dot product w1*x1 + w2*x2 +......wn*xn
    for(size_t i=0; i<features.size(); ++i){
        // adds bias to the sum z = wx+b
        result += weights[i] * features[i];
    }
    return result;  //return the yhat predicted value
}

/*

*/