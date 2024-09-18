from softmax_class import Softmax
from relu_class import ReLU

class Activation_Function(ReLU, Softmax):
    def forward_feed(self, choice, input):
        if choice == "Relu":
            result = ReLU.relu_activation(input)
            return result

        elif choice == "Softmax":
            result = Softmax.softmax_activation(input)
            return result
        
        else:
            print("\nTypo. Wrong forward activation function choice selection.")
    
    def jacobian_calc(self, choice, input):
        if choice == "Relu":
            result = ReLU.relu_derivative(input)
            return result

        elif choice == "Softmax":
            result = Softmax.softmax_derivative(input)
            return result

        else:
            print("\nTypo. Wrong backwards activation function choice selection.")
