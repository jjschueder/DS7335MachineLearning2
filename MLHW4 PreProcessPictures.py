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
ximg_array_list = []
img_28X28_list = []
xfar =np.empty((1,28,28,1), dtype=float)
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
        #img = cv2.imread(path + '\\' + name)
        img = Image.open(path + '\\' + name).convert("L")
        img = np.resize(img, (28,28,1))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28,28,1)
        
        xfar = np.append(xfar,im2arr, axis = 0) 


        img_28x28 = np.array(img_pil.resize((28,28), Image.ANTIALIAS))
        img_28X28_list.append(img_28x28)
        ximg_array = (img_28x28.flatten())
        ximg_array  = ximg_array.reshape(-1,1).T
        ximg_array_list.append(ximg_array)
        
xfar = xfar[1:28]
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

def reshape_data(train, num_classes):
#    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
#    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
#    x_train = x_train.astype('float32')
#    x_test = x_test.astype('float32')
#    x_train /= 255
#    x_test /= 255
#    print('x_train shape:', x_train.shape)
#    print(x_train.shape[0], 'train samples')
#    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
#    y_test = keras.utils.to_categorical(test[1], num_classes)
    return y_train

num_classes = 5

yjoesrs = reshape_data((xfar, Yarr), num_classes = num_classes)


