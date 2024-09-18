import numpy as np

class Backpropagation():
    def cost_function(self, num_output_neuron, train_value, label_value):
        cost = (1/num_output_neuron) * np.sum((train_value - label_value)**2)

        return cost

    def gradient_hidden1(self):
        pass
