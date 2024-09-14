import numpy as np

class Activation_Function:
    def __init__(self):
        self.output = []
    
    def relu(self, input):
        for i in input:
            self.output.append(max(0, i))
    
    def softmax(self, input):
        denominator = sum(np.exp(input))
        for i in input:
            numerator = np.exp(i)
            self.output.append(numerator / denominator)
