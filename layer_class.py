import numpy as np
class Layer_Dense:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.output = None

    def forward_pass(self, input):
        self.output = input @ self.weights + self.biases

    def backpropagation(self):
        pass
