import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_inputs))
        self.output = None

    def forward(self):
        pass

    def backpropagation(self):
        pass
