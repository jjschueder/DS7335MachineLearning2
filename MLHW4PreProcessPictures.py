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
from PIL import Image, ImageOps

from tensorflow import keras
path = r'C:\Users\jjschued\Documents\GitHub\DS7335MachineLearning2\letters for joe-20210318T011401Z-001\letters for joe'


import csv
from pprint import pprint
def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def black_background_thumbnail(source_image, thumbnail_size=(200,200)):
    background = Image.new('RGBA', thumbnail_size, "black")    
    #source_image = Image.open(path_to_image).convert("RGBA")
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    background.paste(source_image, (int((thumbnail_size[0] - w) / 2), int((thumbnail_size[1] - h) / 2) ))
    return background

def get_images():
    
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
    ximg_28X28_list = []
    xfar =np.empty((1,28,28,1), dtype=float)
    for dirpath, dirnames, files in os.walk(path):
        for name in files:
            print(path + '\\' + name)
            totalpath = path + '\\' + name
            for k, v in labeldict.items():
                #add labels of images to list to get to Y values
                if k == totalpath:
                    #print(v)
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
            #https://medium.com/@ashok.tankala/build-the-mnist-model-with-your-own-handwritten-digits-using-tensorflow-keras-and-python-f8ec9f871fd3
            img = Image.open(path + '\\' + name).convert('L')
            #.convert('RGB')
            #plt.imshow(img)
            #.convert("L")
            #img =  make_square(img)
            #img = ImageOps.invert(img)
            #plt.imshow(img)
            #img =  black_background_thumbnail(img)
            #plt.imshow(img)
       
            img_28x28 = np.array(img.resize((28,28), Image.ANTIALIAS))
            #img = np.resize(img, (28,28,1))
            #plt.figure(figsize=(50,50))
            plt.imshow(img_28x28)
            ximg_28X28_list.append(img_28x28)
            #im2arr = np.array(img)
            im2arr = img_28x28.reshape(1,28,28,1)        
            xfar = np.append(xfar,im2arr, axis = 0)
            
            
            
            
#            flippedlr_img = np.fliplr(img_28x28)
#            ximg_28X28_list.append(flippedlr_img)
#            handY.append(label)  
#            im2arr = np.array(flippedlr_img)
#            im2arr = im2arr.reshape(1,28,28,1)        
#            xfar = np.append(xfar,im2arr, axis = 0)
#            
#            flippedud_img = np.flipud(img_28x28)
#            ximg_28X28_list.append(flippedud_img)
#            handY.append(label)  
#            im2arr = np.array(flippedud_img)
#            im2arr = im2arr.reshape(1,28,28,1)        
#            xfar = np.append(xfar,im2arr, axis = 0)
#            
#            left = np.roll(img_28x28, 3)
#            ximg_28X28_list.append(left)
#            handY.append(label)  
#            im2arr = np.array(left)
#            im2arr = im2arr.reshape(1,28,28,1)        
#            xfar = np.append(xfar,im2arr, axis = 0)
#    
#            right = np.roll(img_28x28, -3)
#            ximg_28X28_list.append(right)
#            handY.append(label)  
#            im2arr = np.array(right)
#            im2arr = im2arr.reshape(1,28,28,1)        
#            xfar = np.append(xfar,im2arr, axis = 0)
    
    xfar = xfar[1:len(xfar)]
    
    #convert Y labels to numpy array
    Yarr = np.array(handY)
    xfar = xfar.astype('float32')
#    from keras_preprocessing.image import ImageDataGenerator
#    from matplotlib import pyplot
#    # CREATE MORE IMAGES VIA DATA AUGMENTATION
#    #https://keras.io/api/preprocessing/image/
#    #https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
#    #https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
#    #https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist
#    datagen = ImageDataGenerator(
#        rotation_range=10,  
#        zoom_range = 0.10,  
#        width_shift_range=0.1, 
#        height_shift_range=0.1)
#    datagen.fit(xfar)
#    xfar2 = xfar
#    Yarr2 = Yarr
#    batches = 0
#    for x_batch, y_batch in datagen.flow(xfar, Yarr, batch_size=1):
#	# create a grid of 3x3 images
#        #print(x_batch, y_batch)
#        
#        xfar2 = np.append(xfar2,x_batch, axis = 0)
#        Yarr2 = np.append(Yarr2, y_batch, axis = 0)
#        batches += 1
#        if batches >= 20000:
#            # we need to break the loop by hand because
#            # the generator loops indefinitely
#            break
        

        
    return xfar, Yarr

#X, Y = get_images()
##
##
##print out all images in 28X28 format
##this is essentially our X values
#compact_2828_list = []
#for x in ximg_28X28_list:
#    print("next image:")
#    #print(x[:,:,0])
#    plt.imshow(x)
#    #take out other indexes and create new list
#    #compact_2828_list.append(x[:,:,0])
#    plt.show()
#
#plt.imshow(img)
#plt.show()
#
#img_28x28 = np.array(img.resize((28,28), Image.ANTIALIAS))
##    
#plt.imshow(img_28x28)
#plt.show()
#
#
#np.shape(flippedud_img)
#
#left = np.roll(flippedud_img, 3)
#
#right = np.roll(flippedud_img, -3)
#plt.imshow(left)
#plt.imshow(right)
#
#plt.show()

#whoknows = np.roll(img, -3, axis = 2)
#plt.imshow(whoknows)
