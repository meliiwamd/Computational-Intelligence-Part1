# Q5_graded
# Do not change the above line.

# This cell is for your imports

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Q5_graded
# Do not change the above line.

# This cell is for your codes.

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# These are different activation functions.
# ~Backward are their derivations, used in backpropagation step.

def relu(Z):
  # Z as cache
  A = np.maximum(0, Z)
  return A, Z

def reluBackward(dA, cache):
  Z = cache
  dZ = np.array(dA, copy=True)
  dZ[Z <= 0] = 0
  return dZ

def sigmoid(Z):
  # Z as cache
  A = 1. / (1 + np.exp(-Z))
  return A, Z

def sigmoidBackward(dA, cache):
  Z = cache
  s = 1 / (1 + np.exp(-Z))
  dZ = dA * s * (1 - s)
  return dZ

def softmax(Z):
  # Z as cache
  Z = Z - np.max(Z, axis=0, keepdims=True)
  exponents = np.exp(Z)
  A = exponents / np.sum(exponents, axis=0, keepdims=True)
  return A, Z

def softmaxBackward(dA, cache):
  Z = cache
  s, cache = softmax(Z)
  dZ = dA * s * (1 - s)
  return dZ


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Here x is actually a*v(j), and v(j) = Sigma(w(ji) * y(i))

def calculateV(vector1, vector2):
  return np.dot(vector1, vector2);


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Initialize weights and biases for each layer depending on number of neurons.

def initializeParameters(layers):
  parameters = {}
  for i in range(len(layers) - 1):
    parameters['W' + str(i + 1)] = np.random.randn(layers[i+1], layers[i]) * 0.01
    parameters['b' + str(i + 1)] = np.random.randn(layers[i+1], 1) * 0.01
  
  return parameters


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# The cost.
def computeCost(AL, Y):
  m = Y.shape[1]

  cost = (-1./ m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log( 1-AL)))
  
  return cost

def checkType(y):
  classes, m = y.shape

  # Binary
  if classes == 1:
    return np.greater_equal(y, 0.5, casting='same_kind')

  # Multiclass 
  else:
    return np.argmax(y, axis=0)

def computeAccuracy(AL, y):
  classes, m = y.shape

  if classes == 1:
    accuracy = 100 - np.mean(np.abs(AL - y)) * 100

  else:
    AL_classified = checkType(AL)
    y_classified = checkType(y)
    # Here we check how many answers were predicted right
    num_equals = np.sum(AL_classified == y_classified)
    accuracy = num_equals / m

  return accuracy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# As explanation in the document, we have two kinds of passing the datas.
# Forward passing which is the first step, and we reach the result from output layer.
# Backpropagate the output layer's error to all of the hidden layers.
# Update the weights.


# We start with forward passing.

def linearForward(A, W, b):
  Z = calculateV(W, A) + b
  cache = (A, W, b)
  
  return Z, cache

# Now the pre-activation parameter is passed to an activation function.
# Activation fun ction result

def linearActivationForward(preActivation, W, b, activation):
  # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".

  if activation == "sigmoid":
    
    Z, linear_cache = linearForward(preActivation, W, b)
    A, activation_cache = sigmoid(Z)

  elif activation == "relu":

    Z, linear_cache = linearForward(preActivation, W, b)
    A, activation_cache = relu(Z)

  elif activation == "softmax":

    Z, linear_cache = linearForward(preActivation, W, b)
    A, activation_cache = softmax(Z)
  
  cache = (linear_cache, activation_cache)

  # Results in post-activation value.
  return A, cache

# Do it for all of the layers, calculate post-activation value.
# Until we're in last layer, and have the error in next part of code

def lModelForward(X, parameters):
  caches = []
  A = X
  L = len(parameters) // 2       
  
  for l in range(1, L):
    preActivation = A 
    A, cache = linearActivationForward(preActivation, 
                                          parameters["W" + str(l)], 
                                          parameters["b" + str(l)], 
                                          activation='relu')
    caches.append(cache)

  # It depends on our output layer's number of neurons    
  # Sigmoid if binary
  if parameters["W" + str(L)].shape[0] == 1:
    postActivation, cache = linearActivationForward(A, 
                                          parameters["W" + str(L)], 
                                          parameters["b" + str(L)], 
                                          activation='sigmoid')
  # Softmax if not
  else:
    postActivation, cache = linearActivationForward(A, 
                                          parameters["W" + str(L)], 
                                          parameters["b" + str(L)], 
                                          activation='softmax')
  caches.append(cache)
  return postActivation, caches


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def linearBackward(dZ, cache):
  A_prev, W, b = cache
  m = A_prev.shape[1]

  dW = (1. / m) * np.dot(dZ, cache[0].T) 
  db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
  dA_prev = np.dot(cache[1].T, dZ)

  return dA_prev, dW, db

def linearActivationBackward(dA, cache, activation):
  linear_cache, activation_cache = cache
  
  if activation == "relu":

    dZ = reluBackward(dA, activation_cache)
    dA_prev, dW, db = linearBackward(dZ, linear_cache)
    
  elif activation == "sigmoid":

    dZ = sigmoidBackward(dA, activation_cache)
    dA_prev, dW, db = linearBackward(dZ, linear_cache)
  elif activation == "softmax":

    dZ = softmaxBackward(dA, activation_cache)
    dA_prev, dW, db = linearBackward(dZ, linear_cache)

  return dA_prev, dW, db


def lModelBackward(AL, Y, caches):
   
  grads = {}
  L = len(caches) # the number of layers
  m = AL.shape[1]
  Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

  # Calculate output layer error 
  
  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  
  current_cache = caches[-1]
  
  # Binary one
  if Y.shape[0] == 1 :
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linearActivationBackward(dAL, current_cache, activation="sigmoid")
  
  else:
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linearActivationBackward(dAL, current_cache, activation="softmax")
  
  for l in reversed(range(L-1)):
    current_cache = caches[l]
    
    dA_prev_temp, dW_temp, db_temp = linearActivationBackward(grads["dA" + str(l + 2)], current_cache, activation="relu")
    grads["dA" + str(l + 1)] = dA_prev_temp
    grads["dW" + str(l + 1)] = dW_temp
    grads["db" + str(l + 1)] = db_temp

  return grads


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Updating the weights in order to get closer to the desired answer.

def updateParameters(parameters, grads, learning_rate):
  
  L = len(parameters) // 2 # number of layers in the neural network

  for l in range(L):
    parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
    parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
      
  return parameters


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, per100 = True):#lr was 0.009
    
    np.random.seed(1)
    costs = []                         # keep track of cost
    accuracies = []
    
    # Parameters initialization.
    parameters = initializeParameters(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = lModelForward(X, parameters)
        
        # Compute loss.
        # Compute cost.
        cost = computeCost(AL, Y)

        # Compute accuracy.
        accuracy = computeAccuracy(AL, Y)
        accuracies.append(accuracy)
    
        # Backward propagation.
        grads = lModelBackward(AL, Y, caches)

        # Update parameters.
        parameters = updateParameters(parameters, grads, learning_rate)
                
        # Print the cost and accuracy in each iteration of training example
        if per100:
          if print_cost and i % 100 == 0:
            print ("After iteration %i:" %i, "Cost: " , cost, "Accuracy: " , accuracy) 
            costs.append(cost)
        else:
          if print_cost :
            print ("After iteration %i:" %i, "Cost: " , cost, "Accuracy: " , accuracy) 
            costs.append(cost)
            
    print("Training Data Done!")
    print("**************************************************")

    
    return parameters, accuracies, costs


  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Q5_graded
# Do not change the above line.

# This cell is for your codes.
from keras.utils import np_utils

# Get the training data and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data using numpy

# Our data are pictures (2D -> 28 * 28 pixels) and we convert them into 784 inputs (Assume it as an array)
# We do this reshape operation with numpy, and for both training and test data 
x_train = x_train.reshape(60000, 784).astype('float32').T
x_test = x_test.reshape(10000, 784).astype('float32').T

# Normalize input data valuesin order to result in better and more accurate answers
x_train = x_train / 255.0
x_test = x_test / 255.0

Y_train = np_utils.to_categorical(y_train, 10).T
Y_test = np_utils.to_categorical(y_test, 10).T

learning_rate = 0.2

parameters, accuracies, costs = L_layer_model(x_train, Y_train, [784, 64, 10], learning_rate, 1000, print_cost = True, per100 = True)

# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

# plot the acuuracy
plt.plot(np.squeeze(accuracies))
plt.ylabel('acuuracy')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

# Now time to test our model.
# Here we only need the final answer of forward passing.
# Final post-activation, becuase it shows us the result for test.

# Therefore we only call forward pass.
# Parameters are the model we've trained
# We want to compare Y_test which are the actual results, and post-activations.

postActivationTest, cache = lModelForward(x_test, parameters)
accuracyTest = computeAccuracy(postActivationTest, Y_test)
costTest = computeCost(postActivationTest, Y_test)
print('Accuracy for test data:', accuracyTest)
print('Cost for test data:', costTest)
print("Test Data Done!")
print("**************************************************")



