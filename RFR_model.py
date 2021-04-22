'''
pH-calculator
Heon Joon Lee
This code is used to develop a random forest regression model based on the dataset retrieved via datagenerator.py (csv file),
use 80% data to train the model, 20% to test, then plot the predictive performance of the model compared to y_test.
Along with the scatter plot are mean absolute error (MAE) and mean squared error values (MAE).
This model is also used to predict the pH of an input image for a color within the pH scale. 
'''

from sklearn import metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib


def build_network(x_train, y_train, x_test, y_test):
    '''
    This function holds the neural network model.
    After building the model, the model is then trained, saved, and its performance is outputted.

    **Parameters**
        x_train: 
            The set of training data, which should be 80% from dataset (39123 data points).
        y_train:
            The set of labels for the corresponding training data, which is 80% from dataset.
        x_test:
            The set of testing data, which is 20% from dataset (9781 data points).
        y_test:
            The set of labels for the corresponding testing data, which is 20% from dataset.
    **Returns**
        model:
            A class object that holds the trained model.
    '''
    # Build the model
    model = RandomForestRegressor(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    # Save the model
    joblib.dump(model, "./pH_model.joblib")
    print("Model saved as pH_model.joblib.")
    # calculate MAE and MSE of model
    MAE = metrics.mean_absolute_error(y_test, y_predict)
    print('MAE: ' + str(MAE))
    MSE = metrics.mean_squared_error(y_test, y_predict)
    print('MSE: ' + str(MSE))

    # Plot a scattter graph of pH predicting performance based on y_test and y_predict (output from x_test using model).
    plt.figure()
    plt.xlabel('y_test')
    plt.ylabel('prediction')
    plt.scatter(y_test, y_predict)
    fig = plt.gcf()
    fig.savefig("PH_performance.png")
    return model


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load test/train data into appropriate training and testing variables
    df = pd.read_csv('dataset.csv')
    dataset = df.values
    dataset
    x = dataset[:, 1]
    y = dataset[:, 0]
    # scale the weighted grayscale data into values between 0-1.
    x_scale = x / 255

    # Reserve 80% for training, and 20% for testing. Reshape data into 1x1 arrays.
    x_train, x_test, y_train, y_test = train_test_split(
        x_scale, y, test_size=0.2)
    x_train = x_train.reshape(39123, 1)
    x_test = x_test.reshape(9781, 1)

    # Show dataset if desired
    print("Showing train dataset: ")
    print(x_train, y_train)
    print("Showing test dataset: ")
    print(x_test, y_test)

    # Build network
    build_network(x_train, y_train, x_test, y_test)
