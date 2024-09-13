import numpy as np
from layer_class import Layer_Dense

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward_pass(inputs)
print(layer1.output)

layer2.forward_pass(layer1.output)
print(layer2.output)
