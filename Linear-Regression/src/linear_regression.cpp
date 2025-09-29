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
    if (features.size() != weights.size()) {
        std::cerr << "Feature size mismatch: expected " << weights.size()
                  << ", got " << features.size() << "\n";
        return 0.0;  // or throw an exception
    }
    double result = bias;
    // computing the dot product w1*x1 + w2*x2 +......wn*xn
    for(size_t i=0; i<features.size(); ++i){
        // adds bias to the sum z = wx+b
        result += weights[i] * features[i];
    }
    return result;  //return the yhat predicted value
}

/*============
fit() function implements batch gradient descent
- computes the predictino for each data point
- compute the error : error = yhat - y
- accumulate gradients for weights and bias
- update weights and bias using learning rate
- prints mean squared error every 100 epochs for monitoring
=============
*/

void LinearRegression::fit(const std::vector<DataPoint> &dataset, double lr, int epochs){
    int n = dataset.size(); // no. of datapoints
    for(int epoch=0; epoch<epochs; ++epoch){
        // using vec to compute gradient for each feature separately
        std::vector<double> weightGrad(weights.size(), 0.0);    // storing accumulated gradients for each weight
        double biasGrad = 0.0;      //storing accumulated gradient for the bias

        // looping over the dataset
        // using the reference to avoid copying each datapoint
        for(const auto& point: dataset){
            if (point.features.size() != weights.size()) {
                std::cerr << "Bad datapoint: expected " << weights.size()
                        << " features, got " << point.features.size() << "\n";
                continue;  // skip this row
            }
            double pred = predict(point.features);
            double error = pred - point.target;     // error = yhat - y

            // compute weight gradients
            for(size_t i=0; i<weights.size(); ++i){
                //dfmse/dw = (1/n) * summation of[(yhat - y) * x[i]]
                weightGrad[i] += error * point.features[i];
            }
            // dfmse/db = (1/n) * summation of[yhat - y]
            biasGrad += error;
        }

        // updating each weight using gradient descent 
        // wi = wi - lr * dloss/dwi
        for(size_t i=0; i<weights.size(); ++i){
            //weights[i] = weights[i] - lr * weightGrad[i] / n;
            weights[i] -= lr * weightGrad[i] / n;
        }
        bias -= lr * biasGrad / n;

        if(epoch % 100 == 0){
            double loss = 0.0;
            for(const auto& point: dataset){
                double e = predict(point.features) - point.target;
                loss += e * e;
            }
            loss = loss /n;
            std::cout<<"Epoch "<<epoch<<", MSE: "<<loss<<"\n";
        }
    }
}
