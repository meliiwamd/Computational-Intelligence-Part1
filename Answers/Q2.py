# Q2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q2_graded
# Do not change the above line.

# This cell is for your codes.

class ActivationFunction:
  def Activate(self, Inputs, Weights, Outputs):
    if (np.dot(Inputs, Weights) > 0 and Outputs == 1):
      return True
    elif (np.dot(Inputs, Weights) < 0 and Outputs == -1):
      return True
    else:
      return False



# Q2_graded
# Do not change the above line.

# This cell is for your codes.
# It's almost like the prevoius question

Weights = np.array([0.1, 0.2, 0.4])
Etha = 0.1
Activation = ActivationFunction()


# Read data.txt file to extract Input and Output array
DataSet = open('/content/data.txt', 'r')
Samples = DataSet.readlines()

InputsArray = []
OutputsArray = []
XListLeft = []
YListLeft = [] 
XListRight = []
YListRight = [] 


# Split by ,
for X in Samples:
  Splitted = X.split(",")
  x = float(Splitted[0])
  y = float(Splitted[1])
  InputsArray.append([x, y, 1])
  if float(Splitted[2]) == 0:
    OutputsArray.append(-1)
    XListLeft.append(x)
    YListLeft.append(y)
  else:
    OutputsArray.append(1)
    XListRight.append(x)
    YListRight.append(y)
  
  


# Change the arrays to Numpy
Inputs = np.array(InputsArray)
Outputs = np.array(OutputsArray)

# Errors array
ErrorList = []

Changed = 20 
while Changed > 0.0000000001:
  ErrorInEachIteration = 0
  for x in range(len(Samples)):
    functionOutput = np.dot(Weights, Inputs[x])
    Error = Outputs[x] - functionOutput
    Multiplied = Etha * Error * Inputs[x]
    Weights = (Weights + Multiplied) 

    # Normalize
    Weights /= np.min(abs(Weights))

    Changed = np.max(abs(Multiplied))
    Etha = Etha / 1.00001

    # Activation function
    if (Activation.Activate(Inputs[x], Weights, Outputs[x]) == False):
      ErrorInEachIteration += 1

  ErrorList.append(ErrorInEachIteration)

print('Weights : ', Weights)
# Plot the error of each iteration
ErrorNumpy = np.array(ErrorList)

# naming the x axis
plt.xlabel('Error in each iteration')
# naming the y axis
plt.ylabel('Number of failures')

plt.plot(ErrorNumpy)
plt.show()

# Plot points


XPointsLeft = np.array(XListLeft)
YPointsLeft = np.array(YListLeft)

XPointsRight = np.array(XListRight)
YPointsRight = np.array(YListRight)

plt.plot(XPointsLeft, YPointsLeft, 'o')
plt.plot(XPointsRight, YPointsRight, '^g')

# Plot the final result
x = np.linspace(-220, -60, 300)
y = x * 1.3877228 + 379.42755013
# Create the plot
# naming the x axis
plt.xlabel('Binary classification')
# naming the y axis
plt.ylabel('Decision boundary')

plt.plot(x, y)
  
# Show the plot
plt.show()

