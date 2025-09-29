## Linear Regression in C++ for Iris Dataset Classification
### Overview

This project implements a Linear Regression model in C++ and applies it to the Iris dataset. The goal is to learn the fundamentals of:
- Reading CSV files in C++

- Handling datasets with features and targets

- Implementing Linear Regression from scratch

- Performing gradient descent

- Basic feature scaling

- Predicting and evaluating outputs

⚠️ Note: Linear Regression is naturally a regression algorithm, not a classification algorithm. Applying it to classification problems (like Iris) will produce approximate predictions. For true classification, Logistic Regression or Softmax is recommended.

### Project Structure

```bash
Linear-Regression/
├─ data/
│  └─ iris.csv               # Iris dataset
├─ include/                  # Header files
│  ├─ data_point.h           # DataPoint struct for features and target
│  ├─ csv_loader.h           # Function declaration for CSV loading
│  └─ linear_regression.h    # LinearRegression class declaration
├─ src/
│  ├─ main.cpp               # Main program to load data, train model, and test predictions           
│  ├─ csv_loader.cpp         # CSV loading implementation  
│  └─ linear_regression.cpp  # LinearRegression implementation (fit, predict)
├─ README.md                 # Project documentation
```
### Code Explanation
1. **data_point.h**

Defines a simple DataPoint struct:
```python
struct DataPoint {
    std::vector<double> features; // feature values
    double target;                // target value (class label or numeric)
};
```
- features stores numeric values for each input dimension.

- target stores the output (numeric for regression, or encoded class for classification).

2. **csv_loader.h** and **csv_loader.cpp**

- Reads CSV files and converts each row into a DataPoint.

- Handles:

- Skipping header row

- Trimming whitespaces

- Converting numeric strings to double using std::stod

- Mapping string class labels (like "Setosa") to integers

- Returns a std::vector<DataPoint> for further processing.
```python
std::vector<DataPoint> loadCSV(const std::string &filename, int numFeatures, int targetColumn);
```
- numFeatures: number of numeric features in the dataset.

- targetColumn: index of the target column.

3. **linear_regression.h** and **linear_regression.cpp**

- Implements Linear Regression:

- Constructor: initializes weights and bias to 0

- predict(): computes output y_hat = w1*x1 + w2*x2 + ... + bias

- fit(): batch gradient descent

- Computes prediction error

- Updates weights and bias

- Prints Mean Squared Error (MSE) every 100 epochs
```python
class LinearRegression {
private:
    std::vector<double> weights;
    double bias;
public:
    LinearRegression(int numFeatures);
    double predict(const std::vector<double>& features) const;
    void fit(const std::vector<DataPoint>& dataset, double lr, int epochs);
};
```

4. **main.cpp**

- Loads CSV dataset

- Scales features to [0,1] to improve gradient descent stability

- Scales targets to [0,1] for classification hack

- Initializes LinearRegression with the correct number of features

- Trains the model using fit()

- Predicts outputs for the first few samples and prints them
```python
for(auto &dp : dataset){
    dp.target /= 2.0;  // 0,1,2 -> 0,0.5,1
}
```
- Predictions converted back to class:
```python
int predClass = std::round(model.predict(dp.features) * 2.0);
predClass = std::max(0, std::min(2, predClass));
```

### Features

- CSV Loading: Supports headers, trims whitespace, handles numeric conversion, and maps string labels to integers.

- Feature Scaling: Normalize features for gradient descent stability.

- Linear Regression from Scratch: No external ML libraries required.

- Training with Batch Gradient Descent: Supports learning rate and epochs configuration.

- Prediction: Supports numerical prediction and approximate class prediction for classification tasks.

### Notes / Limitations

- Linear Regression is not ideal for classification:

- Predictions are continuous; rounding is used to approximate class labels.

- Model may predict blocks of the same class if dataset is ordered.

- Iris Dataset Example:

- 4 numeric features: sepal length, sepal width, petal length, petal width

-- Target: "Setosa"=0, "Versicolor"=1, "Virginica"=2

- For real classification, consider:

-- Logistic Regression (binary)

-- Softmax Regression (multi-class)

-- One-hot encoding + Linear Regression per class
