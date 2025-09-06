// Implementing small scale LR model with custom small data points

#include <stdio.h>
#include <math.h>

// func to calculate predictions -> yhat
float predict(float w, float b, float x){
    return x*w + b;
}

// function to calculate mean squared error
float mean_squared_error(float x[], float y[], float w, float b,int n){
    float error_sum = 0.0;
    for(int i=0; i<n;i++){
        float y_pred = predict(w, b, x[i]);
        float diff = y[i] - y_pred;
        error_sum += diff * diff;       // because square of (y[i] - y_pred)^2
    }
    return error_sum/n;     // taking mean
}

// func to train model using gradient descent
void linear_reg(float x[], float y[], int n, float lrate, int epochs){
    float w=0.0, b=0.0;   //intial parameters

    // for every epoch
    for(int epoch=0; epoch<epochs; epoch++){
        float gw = 0.0, gb = 0.0;

        //compute gradients
        for(int i=0; i<n; i++){
            float y_pred = predict(w, b, x[i]);
            float error = y[i] - y_pred;
            gw +=  -2* x[i] * error;
            gb += -2 * error;
        }
        gw = gw /n;
        gb = gb /n;

        // Update parameters
        w -= lrate * gw;
        b -= lrate * gb;

        // Print progress every 100 epochs
        if (epoch % 100 == 0) {
            float mse = mean_squared_error(x, y, w, b, n);
            printf("Gradient w -> %.4f\tGradient b -> %.4f\n", gw, gb);
            printf("Epoch %d -> w=%.4f,\t b=%.4f,\t MSE=%.4f\n", epoch, w, b, mse);
        }
    }
    printf("\nFinal trained model: y = %.4fx + %.4f\n", w, b);
}

int main(){
    // user data {small for now}
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {2.5, 4.2, 6.0, 7.8, 10.1};    
    int n = sizeof(x) / sizeof(x[0]);   // n = 5* 4 / 4 => 5 

    float learning_rate = 0.001;
    int epochs = 1000;

    linear_reg(x, y, n, learning_rate, epochs);
    return 0;
}