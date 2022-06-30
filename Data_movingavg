''' This code is used to apply moving average to data collected in a csv file
    I normally have them all put into a csv file to save a raw data file for reviewers or readers for publications. '''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

with open('stdev.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerow(["swelling"])

    data = pd.read_csv('singlegel.csv')
    t = data['Time']
    r = data['s2']

    x = np.array(t)
    y0 = np.array(r)

    def moving_average(x, y):
        return np.convolve(x, np.ones(y), 'valid') / y

    new_y0 = moving_average(y0, 8)
 

    for i in new_y0:
        y0_list = i

        csv_output.writerow([y0_list])
    
