# pH calculator
Heon Joon Lee (hlee260@jhmi.edu) EN 540.635 Software Carpentry

The goal of this project is to use Python to generate a neural network that could identify the RGB values from a specific color (e.g. indicated using pH paper) and calculate the pH of that specific color using the model. To design this set of code we need to:

# Requirements
Python 3
Install libraries: Keras, Tensorflow, PIL, skscikit-learn, os, pandas, matplotlib, numpy, joblib.
A collection of images for colors of varying pHs (from a pH scale, for instance). These images should be labeled with specific pH values.

# Workflow
1) Using datagenerator.py RGB values of each pH color are retreived from a pre-existing pH scale (.png). Specifically, we cropped out small images of each color from the scale, and generated their RGB values, calculated the weighted grayscale value (c = 0.299 * R + 0.587 * G + 0.114 * B). These are organized into a csv file (dataset.csv) in order to train/test the model. 
2) Use the find command in order to replace the text .png with nothing. This way we have only the pH values and their corresponding grayscale values.
3) We use pHcalculator.py to build the neural network and predict the pH value of an input image. 80% of the dataset.csv data is used for training the neural network, while the remaining 20% is used for testing. 
4) We build a neural network model based on random forest regression. This constructs a multitude of decision trees and outputs the classifcation or average prediction of each tree. The model is saved as a joblib file (pH_model.joblib) and will be used to predict pH of colors later on.
5) We retreive predicted y values (pH's) based on x_test, and compare with the pre-defined y_test using a scatter plot.
6) We can also retreive the mean absolute error (MAE) and mean squared error values (MAE) for our model.
7) We use our code to input an image file, calculate the weighted grayscale after getting RGB values of a single pixel sample, and use the model to predict the pH of the image.

# Output:
In [1]: run phcalc_rfr.py

printing train dataset: 

[[0.16840261]
 [0.17898431]
 [0.12039739]
 ...
 [0.10928889]
 [0.21052941]
 [0.16117778]] 
 
 [10.  1.  0. ... 11.  5.  7.]
 
 printing test dataset: 
 
[[0.12039739]
 [0.28255294]
 [0.24054641]
 ...
 [0.11503529]
 [0.11503529]
 [0.17221699]]
 
 [ 0.  3.  4. ... 13. 13.  9.]
 
 Model saved.

MAE: 0.00095624643043381

MSE: 0.0029855053851068074

pH prediction:

5.5.png, [5.43535556]

5.3.png, [5.25073928]

# Scatter plot of model performance:

![PH_performance_example](https://user-images.githubusercontent.com/82513993/115661940-422b4880-a30c-11eb-8e71-67c570d59da9.png)
