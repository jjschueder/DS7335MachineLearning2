# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:11:26 2021

@author: jjschued
"""

#Below you see a tutorial from Keras on using transfer learning.  They 
#train their models on have the digits and predict the second half.   
#Your homework is to train on all digits and make your own handwritten data 
#set of 5 characters (ie A, B, C, D, E)  and transfer your minist trained model 
#over to them.  Enjoy!


#Code From https://keras.io/examples/mnist_transfer_cnn/
from __future__ import print_function
import datetime
from tensorflow import keras
from tensorflow.keras.datasets import mnist
#data = keras.datasets.mnist.load_data(path="mnist.npz")

from tensorflow.keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D

from tensorflow.keras import backend as K

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

now = datetime.datetime.now

from MLHW4PreProcessPictures import get_images


#######################################################################
##Below is based on office hours. Above was from assignment handout
##
#######################################################################

#Links: 
#https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53    

#######################################################################
#Show how learning is looking
#######################################################################
def plot_training_curves(history, title=None):
    ''' Plot the training curves for loss and accuracy given a model history
    '''
    # find the minimum loss epoch
    minimum = np.min(history.history['val_loss'])
    min_loc = np.where(minimum == history.history['val_loss'])[0]
    # get the vline y-min and y-max
    loss_min, loss_max = (min(history.history['val_loss'] + history.history['loss']),
                          max(history.history['val_loss'] + history.history['loss']))
    acc_min, acc_max = (min(history.history['val_accuracy'] + history.history['accuracy']),
                        max(history.history['val_accuracy'] + history.history['accuracy']))
    # create figure
    fig, ax = plt.subplots(ncols=2, figsize = (15,7))
    fig.suptitle(title)
    index = np.arange(1, len(history.history['accuracy']) + 1)
    # plot the loss and validation loss
    ax[0].plot(index, history.history['loss'], label = 'loss')
    ax[0].plot(index, history.history['val_loss'], label = 'val_loss')
    ax[0].vlines(min_loc + 1, loss_min, loss_max, label = 'min_loss_location')
    ax[0].set_title('Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    # plot the accuracy and validation accuracy
    ax[1].plot(index, history.history['accuracy'], label = 'accuracy')
    ax[1].plot(index, history.history['val_accuracy'], label = 'val_accuracy')
    ax[1].vlines(min_loc + 1, acc_min, acc_max, label = 'min_loss_location')
    ax[1].set_title('Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    plt.show()

def reshape_data(train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    return x_train, x_test, y_train, y_test


num_classes=10
filters=32
pool_size=2
kernel_size=3
dropout=0.2
input_shape = (28,28,1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
xrs_train, xrs_test, yrs_train, yrs_test = reshape_data((x_train, y_train),
            (x_test, y_test), num_classes)

#model.summary()
#
#keras.utils.plot_model(model, show_shapes=(True), dpi = 48)

# Look at some random numbers
X_shuffle = shuffle(x_train.copy(), random_state=42)

print('Look at some numbers\n')
plt.figure(figsize = (12,10))
row, colums = 4, 4
for i in range(16):
    plt.subplot(colums, row, i+1)
    plt.imshow(X_shuffle[i].reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()


model = Sequential([
      # convolutional feature extraction
      # ConvNet 1
      keras.layers.Conv2D(filters, kernel_size, padding = 'valid',
              activation='relu',
              input_shape=input_shape),
      keras.layers.MaxPooling2D(pool_size=pool_size),

      # ConvNet 2
      keras.layers.Conv2D(filters, kernel_size,
              padding = 'valid',
              activation='relu'),
      keras.layers.MaxPooling2D(pool_size=pool_size),

#      # ConvNet 3
#      keras.layers.Conv2D(filters, kernel_size,
#              padding = 'valid',
#              activation='relu'),
#      keras.layers.MaxPooling2D(pool_size=pool_size),

      # classification 
      # will retrain from here
      keras.layers.Flatten(name='flatten'),

      keras.layers.Dropout(dropout),
      keras.layers.Dense(128, activation='relu'),
      
      keras.layers.Dropout(dropout, name='penult'),
      keras.layers.Dense(num_classes, activation='softmax', name='last')
  ])





es = keras.callbacks.EarlyStopping(min_delta=0.01, patience=2)
#opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy",
        #loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

history = model.fit(xrs_train, y_train,
                    #validation_data=(xrs_test, yrs_test),
                    validation_split=0.2,
                    batch_size=32,
                    epochs=1000,
                    callbacks=[es])
model.summary()

plot_training_curves(history)

preds = model.predict(xrs_test)
num_classes=10
preds = np.argmax(preds, axis=1).astype("uint8")
predsy = keras.utils.to_categorical(preds, num_classes)
accuracy_score(yrs_test, predsy)

#https://keras.io/guides/transfer_learning/




layer_trainable = False
for layer in model.layers:
  layer.trainable = layer_trainable
  if layer.name == 'flatten':
    layer_trainable = True

print(f"{'Layer Name':17} {'Is Trainable?'}")
for layer in model.layers:
  print(f"{layer.name:17} {layer.trainable}")


layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(df)        

# get the penultimate layer of the model
penult_layer = model.get_layer(name='penult')

# create a new output layer
output_layer = keras.layers.Dense(5, activation='softmax')(penult_layer.output)

# ConvNet 3
#conv3 = keras.layers.Conv2D(filters, kernel_size,
#                            padding = 'valid',
#                            activation='relu')
#pool3 = keras.layers.MaxPooling2D(pool_size=pool_size)
new_model = Model(model.input, output_layer)
#new_model = Model(inputs=[model.input, conv3, pool3], outputs=output_layer)
new_model.summary()


new_model.compile(
        loss = "sparse_categorical_crossentropy",
        #loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


      


num_classes = 5


#get my images that I process via MLHW4PreProcessPictures.py
Xoi,Yoi = get_images()
(xl_train, xl_test, yl_train, yl_test) = train_test_split(Xoi,Yoi, test_size=0.15, random_state=101)

(xlrs_train, xlrs_test, ylrs_train, ylrs_test) = train_test_split(Xoi,Yoi, test_size=0.15, random_state=101)
num_classes
#ylrs_train = keras.utils.to_categorical(ylrs_train, num_classes)
#ylrs_test = keras.utils.to_categorical(ylrs_test, num_classes)

#(unique, counts) = np.unique(ylrs_train, return_counts=True)
#frequencies = np.asarray((unique, counts))
#
#(unique, counts) = np.unique(ylrs_test, return_counts=True)
#frequencies2 = np.asarray((unique, counts))

xlrs_train= xlrs_train.astype('float32')
xlrs_test = xlrs_test.astype('float32')
xlrs_train /= 255
xlrs_test /= 255

def data_peek(data):
    # Look at some random numbers
    X_shuffle = shuffle(data.copy(), random_state=42)

    print("Look at some random images\n")
    plt.figure(figsize=(12, 10))
    row, colums = 4, 4
    for i in range(16):
        plt.subplot(colums, row, i + 1)
        plt.imshow(X_shuffle[i].reshape(28, 28), interpolation="nearest", cmap="Greys")
    return plt.show()

data_peek(xlrs_train)

#no longer required as shaped in include program
#xlrs_train, xlrs_test, ylrs_train, ylrs_test = reshape_data((xl_train, yl_train),
#            (xl_test, yl_test), num_classes)
es = keras.callbacks.EarlyStopping(min_delta=0.0001, patience=500)
new_model_hist = new_model.fit(xlrs_train, ylrs_train,
                              validation_data=(xlrs_test, ylrs_test),
                              validation_split=0.2,
                              batch_size=32,
                              epochs=1000,
                              callbacks=[es])

#first_array=np.array(xl_test[10])
#first_array= first_array.reshape(-1,28,28,1)
#first_array= first_array.reshape(28,28)
##Not sure you even have to do that if you just want to visualize it
##first_array=255*first_array
##first_array=first_array.astype("uint8")
#plt.imshow(first_array)
##Actually displaying the plot if you are not in interactive mode
#plt.show()

# transfer: train dense layers for new classification task [5..9]
#train_model(model,
#            (xl_train, yl_train),
#            (xl_test, yl_test), num_classes)

plot_training_curves(new_model_hist)

preds = new_model.predict(xlrs_test)

preds = np.argmax(preds, axis=1).astype("uint8")
predsy = keras.utils.to_categorical(preds, num_classes)
acylrs_test = keras.utils.to_categorical(ylrs_test, num_classes)
accuracy_score(acylrs_test, predsy)


#first_array=np.array(xl_test[1])
##Not sure you even have to do that if you just want to visualize it
#first_array= first_array.reshape(-1,28,28,1)
#first_array= first_array.reshape(28,28)
#plt.imshow(first_array)
#
##Actually displaying the plot if you are not in interactive mode
#plt.show()
#
#
#
######################################################
#
#
#
#
#
## BUILD CONVOLUTIONAL NEURAL NETWORKS
##https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
#nets = 3
#modelk = [0] *nets
#
#for j in range(3):
#    modelk[j] = Sequential()
#    modelk[j].add(keras.layers.Conv2D(24,kernel_size=5,padding='same',activation='relu',
#            input_shape=(28,28,1)))
#    modelk[j].add(keras.layers.MaxPooling2D())
#    if j>0:
#        modelk[j].add(keras.layers.Conv2D(48,kernel_size=5,padding='same',activation='relu'))
#        modelk[j].add(keras.layers.MaxPool2D())
#    if j>1:
#        modelk[j].add(keras.layers.Conv2D(64,kernel_size=5,padding='same',activation='relu'))
#        modelk[j].add(keras.layers.MaxPooling2D(padding='same'))
#    modelk[j].add(keras.layers.Flatten(name='flatten'))
#    modelk[j].add(keras.layers.Dense(256, activation='relu'))
#    modelk[j].add(keras.layers.Dropout(dropout, name='penult')),
#    modelk[j].add(keras.layers.Dense(10, activation='softmax'))
#    modelk[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#
#es = keras.callbacks.EarlyStopping(min_delta=0.01, patience=2)
#
#mk3 = modelk[2].fit(xrs_train, yrs_train,
#                    #validation_data=(xrs_test, yrs_test),
#                    validation_split=0.2,
#                    batch_size=32,
#                    epochs=1000,
#                    callbacks=[es])
#
#
#layer_trainable = False
#for layer in modelk[2].layers:
#  layer.trainable = layer_trainable
#  if layer.name == 'flatten':
#    layer_trainable = True
#
#print(f"{'Layer Name':17} {'Is Trainable?'}")
#for layer in modelk[2].layers:
#  print(f"{layer.name:17} {layer.trainable}")
#
#
#layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
#df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
#print(df)        
#
## get the penultimate layer of the model
#penult_layer = modelk[2].get_layer(name='penult')
#
## create a new output layer
#output_layer = keras.layers.Dense(5, activation='softmax')(penult_layer.output)
#
#new_model = Model(modelk[2].input, output_layer)
#new_model.summary()
#
#new_model.compile(loss='categorical_crossentropy',
#                  optimizer='adam',
#                  metrics=['accuracy'])


####to do change alteryx to get A to E instead of B to F..####
##############################################################!

##this was using data from mnist removed as per assignment requirement
#dataset = np.loadtxt('Q:\A_e HandwrittenData.csv', delimiter=',', max_rows = 2000000)        
#dataset = np.delete(dataset, (0), axis=0)
#X = dataset[:,0:784]
#Y = dataset[:,0]
##
#(xl1_train, xl1_test, yl1_train, yl1_test) = train_test_split(X, Y, test_size=0.75, random_state=101)
#yl1_train = yl1_train -1
#yl1_test = yl1_test -1
#
#xl1rs_train, xl1rs_test, yl1rs_train, yl1rs_test = reshape_data((xl1_train, yl1_train),
#            (xl1_test, yl1_test), num_classes)
#
#
#
#es = keras.callbacks.EarlyStopping(min_delta=0.001, patience=200)
#new_model_hist = new_model.fit(xl1rs_train, yl1rs_test,
#                              #validation_data=(xlrs_test, ylrs_test),
#                              validation_split=0.2,
#                              batch_size=32,
#                              epochs=1000,
#                              callbacks=[es])
#
#preds = new_model.predict(xl1rs_test)
#
#preds = np.argmax(preds, axis=1).astype("uint8")
#predsy = keras.utils.to_categorical(preds, num_classes)
#accuracy_score(yl1rs_test, predsy)
#https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
#https://github.com/dipanjanS/hands-on-transfer-learning-with-python/blob/master/notebooks/Ch05%20-%20Unleash%20the%20Power%20of%20Transfer%20Learning/CNN%20with%20Transfer%20Learning.ipynb
####This part loads in sample letters A to E which we will use to perform
# transfer learning on
#########################################################################

#print("Number of layers in the base model: ", len(model.layers))
#fine_tune_at = len(model.layers) - 1
##model.layers[0].trainable = True
## Freeze all the layers before the `fine_tune_at` layer
#for layer in model.layers[1:fine_tune_at]:
#  layer.trainable =  False
#
#num_classes = 5
#
###change class size!?
#######################
##model.layers[8].units = 5
##########################
#model.add(keras.layers.Dense(num_classes, activation= 'softmax', name='lastletters'))
#model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
#              #optimizer = keras.optimizers.RMSprop(lr=base_learning_rate/10),
#              optimizer = 'adam',
#              metrics=['accuracy'])
#  
#model.summary()
##to do adjust which ones I want trainable??
#set_trainable = False
#for layer in model.layers:
#    print(layer.name)
#    if layer.name in ['block5_conv1', 'block4_conv1']:
#        set_trainable = True
#    if set_trainable:
#        layer.trainable = True
#    else:
#        layer.trainable = False