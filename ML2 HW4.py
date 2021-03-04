# -*- coding: utf-8 -*-

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

from tensorflow.keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D

from tensorflow.keras import backend as K
now = datetime.datetime.now
batch_size = 128
num_classes = 5
epochs = 5
img_rows, img_cols = 28, 28
filters = 32
pool_size = 2
kernel_size = 3
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def train_model(model, train, test, num_classes):
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
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]
x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5




feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# create complete model
model = Sequential(feature_layers + classification_layers)

# train model for 5-digit classification [0..4]
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

# transfer: train dense layers for new classification task [5..9]
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)






model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


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

x_train, x_test, y_train, y_test = reshape_data((x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)
t = now()
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
print('Training time: %s' % (now() - t))
score = model.evaluate(x_test, y_test, verbose=0)   

import matplotlib.pyplot as plt
first_array=x_test_gte5[1]
#Not sure you even have to do that if you just want to visualize it
#first_array=255*first_array
#first_array=first_array.astype("uint8")
plt.imshow(first_array)
#Actually displaying the plot if you are not in interactive mode
plt.show()

import csv
rows = [] 
with open('Q:\A_Z HandwrittenData.csv', 'r') as csvfile: 
    # creating a csv reader object 
    csvreader = csv.reader(csvfile) 
      
    # extracting field names through first row 
    fields = next(csvreader) 
  
    # extracting each data row one by one 
    ctr = 0
    for row in csvreader: 
        ctr +=1
        rows.append(row) 
        if ctr >= 200:
            break
import numpy as np  
from sklearn.model_selection import train_test_split      
dataset = np.loadtxt('Q:\A_Z HandwrittenData.csv', delimiter=',', max_rows = 20000)        

X = dataset[:,0:784]
Y = dataset[:,0]

(x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.75, random_state=101)

x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float')
x_test = x_test.reshape(x_test.shape[0], 28, 28).astype('float')

import matplotlib.pyplot as plt
first_array=x_train[9]
#Not sure you even have to do that if you just want to visualize it
#first_array=255*first_array
#first_array=first_array.astype("uint8")
plt.imshow(first_array)
#Actually displaying the plot if you are not in interactive mode
plt.show()
