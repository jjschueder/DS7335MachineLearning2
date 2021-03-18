# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:22:54 2021

@author: jjschued
"""

import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

path = r'C:\Users\jjschued\Documents\GitHub\DS7335MachineLearning2\letters for joe-20210318T011401Z-001\letters for joe'


import csv
from pprint import pprint

with open(r'C:\Users\jjschued\Documents\GitHub\DS7335MachineLearning2\ImageLabels.csv') as file:
    reader = csv.reader(file)
    res = list(map(tuple, reader))

#create label dictionary
labeldict = dict(res)
#path = path + '\canvas (1).png'

#intialize label list
handY  = []

#intialize list of image arrays
img_array_list = []
img_28X28_list = []
for dirpath, dirnames, files in os.walk(path):
    for name in files:
        print(path + '\\' + name)
        totalpath = path + '\\' + name
        for k, v in labeldict.items():
            #add labels of images to list to get to Y values
            if k == totalpath:
                print(v)
                label = v
                if label == 'A':
                    label = 0
                elif label == 'B':
                    label = 1
                elif label == 'C':
                    label = 2
                elif label == 'D':
                    label = 3
                elif label == 'E':
                    label = 4
                handY.append(label)  
        #open image and convert to various np arrays        
        img = cv2.imread(path + '\\' + name)
        imggray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_pil = Image.fromarray(img)
        img_28x28 = np.array(img_pil.resize((28,28), Image.ANTIALIAS))
        img_28X28_list.append(img_28x28)
        img_array = (img_28x28.flatten())
        img_array  = img_array.reshape(-1,1).T
        img_array_list.append(img_array)
        
        
#convert Y labels to numpy array
Yarr = np.array(handY)

#print out all images in 28X28 format
#this is essentially our X values
compact_2828_list = []
for x in img_28X28_list:
    print("next image:")
    #print(x[:,:,0])
    plt.imshow(x[:,:,0])
    #take out other indexes and create new list
    compact_2828_list.append(x[:,:,0])
    plt.show()
    


#first_array=np.array(imggray) 
##Not sure you even have to do that if you just want to visualize it
#first_array=255*first_array
#first_array=first_array.astype("uint8")
#first_array= first_array.reshape(-1,28,28,1)
#first_array= first_array.reshape(28,28)
#
#plt.imshow(first_array)
