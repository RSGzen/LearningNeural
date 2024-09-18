from softmax_class import Softmax
from relu_class import ReLU

class Activation_Function(ReLU, Softmax):
    def forward_feed(self, choice):
        if choice == "Relu":
            result = ReLU.relu_activation()
            return result

        elif choice == "Softmax":
            result = Softmax.softmax_activation()
            return result
        
        else:
            print("\nTypo. Wrong forward activation function choice selection.")
    
    def jacobian_calc(self, choice):
        if choice == "Relu":
            result = ReLU.relu_derivative()
            return result

        elif choice == "Softmax":
            result = Softmax.softmax_derivative()
            return result

        else:
            print("\nTypo. Wrong backwards activation function choice selection.")
