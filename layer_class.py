import numpy as np
from activation_function.activation_func_class import Activation_Function
from backpropagation import Backpropagation
class Layer_Dense():
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weights = np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.output = None

    def forward_pass_net(self, input):
        self.output = input @ self.weights + self.biases
    
    def forward_pass_output(self, input, activation_func):
        temp_activation = Activation_Function()
        if activation_func == "Relu":
            result = temp_activation.forward_feed("Relu", input)
            return result

        elif activation_func == "Softmax":
            result = temp_activation.forward_feed("Softmax", input)
            return result
        
        else:
            print("\nTypo. Wrong activation function selection.")
