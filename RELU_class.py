class ReLU:
    def __init__(self):
        self.output = []
    
    def relu_activation(self, input):
        for i in input:
            self.output.append(max(0, i))
