#include "csv_loader.h"
#include "linear_regression.h"
#include <iostream>
#include <cmath>

int main(){
    std::vector<DataPoint> dataset = loadCSV("../data/iris.csv", 4, 4);
    // initializing a linear reg model with correct number of features
    if (dataset.empty()) {
        std::cerr << "Dataset is empty. Exiting...\n";
        return 1;
    }   
    scaleFeatures(dataset);
    // scale targets to [0,1] for better regression
    for(auto &dp : dataset){
        dp.target /= 2.0;  // 0,1,2 -> 0,0.5,1
    }
    LinearRegression model(dataset[0].features.size());

    //training the model using fit
    model.fit(dataset, 0.05, 2000);

    std::cout<<"\nPredictions for first 80 samples:\n";
    for(int i=0; i<80; i++){
        double y_pred = model.predict(dataset[i].features);
        int predClass = std::round(y_pred * 2.0);    // round to nearest int
        // ensure within 0-2
        predClass = std::max(0, std::min(2, predClass));

        int trueClass = std::round(dataset[i].target * 2.0);
        std::cout << "True: " << trueClass<< ", Predicted: " << predClass << "\n";
    }
    return 0;
}