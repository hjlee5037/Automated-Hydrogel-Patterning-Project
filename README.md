# pH calculator
Heon Joon Lee (hlee260@jhmi.edu) EN 540.635 Software Carpentry

The goal of this project is to use Python to generate a model that could identify the RGB values from a specific color (e.g. indicated using pH paper) and calculate the pH of that specific color using the model.

# Requirements
Python 3
Install libraries: Keras, Tensorflow, PIL, skscikit-learn, os, pandas, matplotlib, numpy, joblib.
A collection of images for colors of varying pHs (from a pH scale, for instance). These images should be labeled with specific pH values.

# Workflow
1) Using datagenerator.py RGB values of each pH color are retreived from a pre-existing pH scale (.png). Specifically, we cropped out small images of each color from the scale, and generated their RGB values, calculated the weighted grayscale value (c = 0.299 * R + 0.587 * G + 0.114 * B). These are organized into a csv file (dataset.csv) in order to train/test the model. 
2) Use the find command in order to replace the text .png with nothing. This way we have only the pH values and their corresponding grayscale values.
3) We use pHcalculator.py to build a machine learning model and predict the pH value of an input image. 80% of the dataset.csv data is used for training the model, while the remaining 20% is used for testing. 
4) We build a machine learning model based on random forest regression. This constructs a multitude of decision trees and outputs the classifcation or average prediction of each tree. The model is saved as a joblib file (pH_model.joblib) and will be used to predict pH of colors later on.
5) We retreive predicted y values (pH's) based on x_test, and compare with the pre-defined y_test using a scatter plot.
6) We can also retreive the mean absolute error (MAE) and mean squared error values (MAE) for our model.
7) We use pH_predict.py open an image of a pH scale (pH_scale.png), calculate the weighted grayscale after getting RGB values of the clicked area, and use the model to predict the pH of the click.

# Output of RFR_model.py:
In [1]: run phcalc_rfr.py

Showing train dataset: 

[[0.3616549 ]
 [0.3663451 ]
 [0.55890588]
 ...
 [0.26102353]
 [0.37072549]
 [0.54860784]] [11.8 11.9  6.6 ... 14.9 13.4  1.6]
 
Showing test dataset: 

[[0.5130902 ]
 [0.54434118]
 [0.52597647]
 ...
 [0.51929804]
 [0.35900392]
 [0.7244902 ]] [ 7.7  9.4  8.1 ...  7.2 13.7  4.6]
 
Model saved as pH_model.joblib.

MAE: 0.00095624643043381

MSE: 0.0029855053851068074

![PH_performance_example](https://user-images.githubusercontent.com/82513993/115661940-422b4880-a30c-11eb-8e71-67c570d59da9.png)

# Output of pH_predict.py
In [2]: run pH_predict.py

<img width="1023" alt="Screen Shot 2021-04-22 at 10 45 43 PM" src="https://user-images.githubusercontent.com/82513993/115810917-c6d59f80-a3bc-11eb-9ab1-8f4c48e6aa3c.png">

pH of clicked region:  [7.66610281]

# References:

1. https://www.javaer101.com/en/article/34524053.html
2. https://www.geeksforgeeks.org/random-forest-regression-in-python/
3. https://medium.com/ampersand-academy/random-forest-regression-using-python-sklearn-from-scratch-9ad7cf2ec2bb
4. https://stackoverflow.com/questions/56787999/python-opencv-realtime-get-rgb-values-when-mouse-is-clicked
5. https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga61d9b0126a3e57d9277ac48327799c80
