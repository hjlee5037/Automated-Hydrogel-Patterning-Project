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
from PIL import Image
import numpy as np
import joblib

def build_network(x_train, y_train, x_test, y_test):
    '''
    This function holds the neural network model.
    After building the model, the model is then trained, saved, and its performance is outputted.

    **Parameters**
        x_train: *numpy*
            The set of training data, which should be 80% from dataset (39123 data points).
        y_train: *numpy*
            The set of labels for the corresponding training data, which is 80% from dataset.
        x_test: *numpy.ndarray*
            The set of testing data, which is 20% from dataset (9781 data points).
        y_test: *numpy.ndarray*
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
    print("Model saved.")
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

def pH_predict(x):
    '''
    This function loads the saved model to predict pH of an input x value.

    **Parameters**
        x_train: *numpy*
            The set of training data.
        y_train: *numpy*
            The set of labels for the corresponding training data.
        x: *numpy.ndarray*
            An input value to predict pH.
    **Returns**
        pH_estimate:
            Estimated pH value.
    '''
    # Load the saved model.
    loaded_model = joblib.load("./pH_model.joblib")
    # Use model to predict pH of input value.
    pH_estimate = loaded_model.predict(np.array([x]).reshape(1, 1))
    return pH_estimate

def recognize_pH():
    '''
    This function retreives RGB from a single pixel of a certain image,
    calculates the weighted grayscale of the image, then inputs the normalized value
    into pH_predict function. The predicted pH value for the image is printed out.
    
    **Parameters**
        x_train: *numpy*
            The set of training data.
        y_train: *numpy*
            The set of labels for the corresponding training data.
        x: *numpy.ndarray*
            An input value to predict pH.
    **Returns**
        pH_estimate:
            Estimated pH value.
    '''
    #input of file name of image with specific pH value and single color.
    filename = '9.7.png'
    img = Image.open(filename)
    # Parallel to x values from data, we calculate weighted grayscale (c) of a single pixel in a image.
    pxl = img.getpixel((1,1))
    c = (((0.299 * int(pxl[0])) + (0.587 * int(pxl[1]))
            + (0.114 * int(pxl[2]))))
    # Normalize the c value between 0 and 1.
    c_norm = int(c) / 255
    #Use pH_predict to calculate pH using model and c_norm.
    predicted_pH = pH_predict(c_norm)
    # Print the file name and predicted pH value according to the model.
    print(f'{filename}, {predicted_pH}')
    return

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
    print("pH prediction: ")
    recognize_pH()
