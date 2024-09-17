import numpy as np

class Softmax():
    def softmax_activation(self, input):
        output = np.empty()

        denominator = sum(np.exp(input))
        for i in input:
            numerator = np.exp(i)
            self.output.append(numerator / denominator)

        return output
