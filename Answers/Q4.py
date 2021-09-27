# Q4_graded
# Do not change the above line.

# This cell is for your imports.

import keras
from keras.layers import Dense
from keras.optimizers import *
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

# Attention: This code is with the help of a video, link below
# https://www.youtube.com/watch?v=fSJxZ_Gc9BM

# But I've totally understand it and declared every section of it in both code, and report 


# Get the training data and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the model we want to use for our neural network
Model = Sequential()

# Our data are pictures (2D -> 28 * 28 pixels) and we convert them into 784 inputs (Assume it as an array)
# We do this reshape operation with numpy, and for both training and test data 
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

# Normalize input data valuesin order to result in better and more accurate answers
x_train = x_train / 255.0
x_test = x_test / 255.0

# Amplify all of the categories we have at the end [0 - 9]
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Adding layers to our neural network model
# The second layer added is actually the last layer and gives us the classifications 
# Here we can add more layers to accelerate the operation (maybe) and increase accurancy
Model.add(Dense(64, input_shape = (784, ), activation = 'relu'))
Model.add(Dense(10, activation = 'softmax'))

# Complie and run the network
Model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["acc"])

# Sumps up the necessary information  
print(Model.summary())

# This part is for test and evaluation, epochs is number of iterations
History = Model.fit(x_train, y_train, epochs = 100, batch_size = 128, validation_data = (x_test, y_test), shuffle = True)

# Draw the error and accuracy diagrams (plot)

# Accuracy
plt.plot(History.history['acc'], 'b')
plt.plot(History.history['val_acc'], 'g')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Error (lose)
plt.plot(History.history['loss'], 'y')
plt.plot(History.history['val_loss'], 'r')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

