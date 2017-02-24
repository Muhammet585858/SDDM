# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/muhammet/.spyder2/.temp.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage

def imageToVector():
    INPUT_FOLDER = 'images/original/'
    OUTPUT_FOLDER = 'images/shrinked/'
    imgAll = os.listdir(INPUT_FOLDER)
    for imgname in imgAll:
        img = misc.imread(INPUT_FOLDER + imgname, 'L')
        #img = ndimage.rotate(img, 45)
        imgShrinked = misc.imresize(img, (5,5)) # just to make it able to run it from home

        print(imgShrinked.flatten())
        print(imgShrinked.flatten('F'))
        print(imgShrinked.ravel())
        misc.imsave(OUTPUT_FOLDER + 'shrinked' + imgname, imgShrinked)
      #  imgShrinked = np.ndarray(imgShrinked)
        np.savetxt("imgArray.csv", imgShrinked.flatten(), delimiter=',', newline=" ")

        
        #when we get stuck when processing in memory. Like that it does not fit. Than change adresses etc.
        


#Should we normalize the data
#What is the degree of rotating

        
        
        