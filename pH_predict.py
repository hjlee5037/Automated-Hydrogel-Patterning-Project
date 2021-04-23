'''
This function is used to open up a pH scale image, and
predict the pH of clicked regions using the model from
RFR_model.py.
'''

import cv2
import joblib
import numpy as np


def pH_predict(c_norm):
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
    pH_estimate = loaded_model.predict(np.array([c_norm]).reshape(1, 1))
    return pH_estimate


def mouse_pH(event, x, y, flags, param):
    '''
    This function identifies x,y positions of click, reads RGB values,
    converts to weighted grayscale, and predicts the pH using pH_predict function.
    **Parameters**
        event:
            Used to define mouse clicking
        x:
            x position of the mouse click.
        y:
            y position of the mouse click.
        flags:
            Indication of how images should be read.
        para:
            Takes in additional variables.

    '''
    if event == cv2.EVENT_LBUTTONDOWN:  # mouse left button down condition
        # retrieve RGB values and calculate weighted grayscale
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        c = (((0.299 * r)) + (0.587 * g)) + (0.114 * b)
    # Normalize the c value between 0 and 1.
        c_norm = int(c) / 255
    # Use pH_predict to calculate pH using model and c_norm.
        predicted_pH = pH_predict(c_norm)
        print("pH of clicked region: ", predicted_pH)


# Read an image, opens a window and bind the mouse_pH function to the window
img = cv2.imread("pH_scale.png")
cv2.namedWindow('mouse_pH')
cv2.setMouseCallback('mouse_pH', mouse_pH)

# Do until esc pressed
while(1):
    cv2.imshow('mouse_pH', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
# if esc pressed, finish.
cv2.destroyAllWindows()
