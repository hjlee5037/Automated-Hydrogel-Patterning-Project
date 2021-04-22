import joblib
from PIL import Image
import numpy as np


def pH_predict(x):
    '''
    This function loads the saved model to predict pH of an input x value.
    **Parameters**
        x:
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


def recognize_pH(filename):
    '''
    This function retreives RGB from a single pixel of a certain image,
    calculates the weighted grayscale of the image, then inputs the normalized value
    into pH_predict function. The predicted pH value for the image is printed out.

    **Parameters**
        filename: *numpy.ndarray*
            An input value to predict pH.
    **Returns**
        pH_estimate:
            Estimated pH value.
    '''
    # input of file name of image with specific pH value and single color.
    img = Image.open(filename)
    # Parallel to x values from data, we calculate weighted grayscale (c) of a single pixel in a image.
    pxl = img.getpixel((1, 1))
    c = (((0.299 * int(pxl[0])) + (0.587 * int(pxl[1]))
          + (0.114 * int(pxl[2]))))
    # Normalize the c value between 0 and 1.
    c_norm = int(c) / 255
    # Use pH_predict to calculate pH using model and c_norm.
    predicted_pH = pH_predict(c_norm)
    # Print the file name and predicted pH value according to the model.
    print(f'{filename}, {predicted_pH}')
    return


filename = '9.7.png'

print("pH prediction:")
recognize_pH(filename)
