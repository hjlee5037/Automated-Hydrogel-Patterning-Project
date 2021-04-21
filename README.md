# pH-calculator
Heon Joon Lee (hlee260@jhmi.edu) EN 540.635 Software Carpentry

The goal of this project is to use Python to generate a calculator that could identify the RGB values from a specific color (e.g. indicated using pH paper) and calculate the pH of that specific color. To design this set of code we need to:

1) RGB values of each pH color are retreived from a pre-existing pH scale (indicated in folder as train_ref.png and test_ref.png for reference pH scales for training and testing the neural network, respectively). Specifically, we cropped out around 33 x 148 (train) or 50 x 94 (test)pixels of each color from the scale and generated each of their RGB files. These are organized into a csv file in order to train/test the model. For train.csv, there are 95443 datasets available, while for test.csv, there are 52657 datasets available.
2) Import the train/test dataset into the phcalc code in order to generate the neural network and train the model. 
3) 


Output:
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

MAE: 0.00095624643043381 <-- This is the mean absolute error

MSE: 0.0029855053851068074 <-- This is the mean squared error

Scatter plot of model performance:
PH_performance.png![PH_performance](https://user-images.githubusercontent.com/82513993/115449874-3dba3f00-a1e9-11eb-8c40-f7744a7f80cd.png)
