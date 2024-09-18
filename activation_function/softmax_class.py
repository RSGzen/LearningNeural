import numpy as np

class Softmax():
    def softmax_activation(self, input):
        output = np.empty()

        denominator = sum(np.exp(input))
        for i in input:
            numerator = np.exp(i)
            self.output.append(numerator / denominator)

        return output
    
    def softmax_derivative(self, input, no_layer):
        output = np.empty()
        for i in range(input):
            if i == no_layer:
                result = np.array([input[i]*(1-input[i])])
                np.append(output, result)
            else:
                result = np.array([-input[i] * input[no_layer]])
                np.append(output, result)
        
        return output
