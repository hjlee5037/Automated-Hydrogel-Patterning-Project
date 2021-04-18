import pandas as pd
import matplotlib.pyplot as plt
import os
from keras.utils import np_utils
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Dropout, Activation


def build_network(x_train, y_train, x_test, y_test, model_name='pH_rgb_calibration.h5'):
    '''
    This function holds the neural network model.
    After building the model, the model is then trained and saved to a .h5
    file.

    **Parameters**

        x_train: *numpy*
            The set of training data, which should be 80% from dataset_x.
        y_train: *numpy*
            The set of labels for the corresponding training data, which is 80% from dataset_y.
        x_test: *numpy.ndarray*
            The set of testing data, which is 20% from dataset_x.
        y_test: *numpy.ndarray*
            The set of labels for the corresponding testing data, which is 20% from dataset_y.
        model_name: *str*
            The filename of the model to be saved.

    **Returns**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.

    '''
    # Build the model
    model = Sequential()
    model.add(Dense(1, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(15))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer='adam')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=128, epochs=20,
                        verbose=2, validation_data=(x_test, y_test))

    # Save the model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Return the model
    return model, history


def plot_performance(model, history, output_name="PH_performance.png"):
    '''
    Retrieve accuracies from the history object and save them to a figure.

    **Parameters**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.
        output_name: *str, optional*
            The filename of the output image.

    **Returns**

        None
    '''
    # Plot accuracy
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    # Calculate loss and accuracy on the testing data
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    # Save figure as .png file
    fig = plt.gcf()
    fig.savefig(output_name)


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load test/train data into appropriate training and testing variables
    df1 = pd.read_csv('dataset_x.csv')
    df2 = pd.read_csv('dataset_y.csv')

    #Reserve 80% for training, and 20% for testing
    x_train, x_test, y_train, y_test = train_test_split(
        df1, df2, test_size=0.2)

    # Normalize the data to help with training
    x_train /= 255
    x_test /= 255

    '''
    Here, we convert classes using one-hot encoding. This means that we
    convert:
    0 -> [1,0,0,0,0,0,0,0,0,0]
    1 -> [0,1,0,0,0,0,0,0,0,0]
    2 -> [0,0,1,0,0,0,0,0,0,0]
    and so on...
    '''
    n_classes = 15
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    # Build network
    model, history = build_network(x_train, y_train, x_test, y_test)

    # Observe the performance of the network
    plot_performance(model, history)
