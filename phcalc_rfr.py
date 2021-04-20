from sklearn import metrics
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def build_network(x_train, y_train, x_test, y_test, output_name="pH_performance_RFR.png"):
    '''
    This function holds the neural network model.
    After building the model, the model is then trained and saves an image file comparing the predicted and tested y (pH) values.

    **Parameters**

        x_train: *numpy*
            The set of training data, which should be 80% from dataset (58574 data points).
        y_train: *numpy*
            The set of labels for the corresponding training data, which is 80% from dataset.
        x_test: *numpy.ndarray*
            The set of testing data, which is 20% from dataset (14644 data points).
        y_test: *numpy.ndarray*
            The set of labels for the corresponding testing data, which is 20% from dataset.
        model_name: *str*
            The filename of the model to be saved.
    **Returns**
        model:
            A class object that holds the trained model.
    '''
    # Build the Random Forest Regressor (RFR) model
    model = RandomForestRegressor(n_estimators = 100, random_state = 1)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    model
    
    #calculate MAE and MSE of model
    MAE = metrics.mean_absolute_error(y_test, y_predict)
    print('MAE: '+str(MAE))
    MSE = metrics.mean_squared_error(y_test, y_predict)
    print('MSE: '+str(MSE))

    plt.figure()
    plt.xlabel('y_test')
    plt.ylabel('prediction')
    plt.scatter(y_test, y_predict)
    fig = plt.gcf()
    fig.savefig(output_name)
    
    # Return the model
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
    x_scale = x / 255

    # Reserve 80% for training, and 20% for testing. Rehsape data for 1 x 1 array.
    x_train, x_test, y_train, y_test = train_test_split(
        x_scale, y, test_size=0.2)
    x_train = x_train.reshape(58574, 1)
    x_test = x_test.reshape(14644, 1)

    # Show dataset if desired
    # print(x_train, y_train)
    # print(x_test, y_test)

    # Build network
    build_network(x_train, y_train, x_test, y_test)
