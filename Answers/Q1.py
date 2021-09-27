# Q1_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1_graded
# Do not change the above line.

# This cell is for your codes.

# We define the initial weights.
Weights = np.array([0.1, 0.2, 0.4])

Inputs = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
Outputs = np.array([1, -1, -1, -1])

Etha = 0.1
Changed = 20


while Changed > 0.000000001:
  for x in range(4):
    functionOutput = np.dot(Weights, Inputs[x])
    Error = Outputs[x] - functionOutput
    Multiplied = Etha * Error * Inputs[x]
    Weights = Weights + Multiplied

    Changed = np.max(abs(Multiplied))
    Etha = Etha / 1.001

# Final weights and last difference
print("Final weights are: ", Weights, "Last error difference is: ", Changed)

  
  






# Q1_graded
# Do not change the above line.

# This cell is for your codes.

