import numpy as np

class Activation_Function:
    def __init__(self):
        self.output = np.empty()
    
    def clear_output(self):
        self.output = np.empty()
    
    def relu(self, input):
        for i in input:
            self.output.append(max(0, i))
    
    def softmax(self, input):
        denominator = sum(np.exp(input))
        for i in input:
            numerator = np.exp(i)
            self.output.append(numerator / denominator)
    
    def tanh(self, input):
        for i in input:
            exp_positive = np.exp(i)
            exp_negative = np.exp(-i)
            result = (exp_positive - exp_negative) / (exp_positive + exp_negative)
            self.output.append(result)
