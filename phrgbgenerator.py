'''
This simple code generates RGB values from references images labeled with a pH value.
This code needs to be in the same folder as all the reference images.
The output will be the file name (0-14.png), and their RGB values in seperate columns.
Once the RGB values of all pH's have been generated, we use excel except to remove ".png".
To do this, we simple use find command and replace .png with nothing to only have the pH values.
'''

from PIL import Image
import glob
import os
import csv

# We need to open a csv file to write the pH and rgb data
with open('o.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)
    csv_output.writerow(["ph", "R", "G", "B"])

    # We need to create a path to the file with png files labeled with pH in the range of 0-14
    for filename in glob.glob("*.png"):
        im = Image.open(filename)
        img_name = os.path.basename(filename)
        pxl = im.load() # generate RGB values

        # Get a tuple of the x and y dimensions of the image
        width, height = im.size
        # Verify all image files labeled with pH in the range of 0-14 have run through the code.
        print("Getting RGB values from reference pH: " + filename)
        
        # Read the details of each pixel and write them to the csv file
        for x in range(width):
            for y in range(height):
                R = pxl[x, y][0]
                G = pxl[x, y][1]
                B = pxl[x, y][2]
                csv_output.writerow([filename, R, G, B])