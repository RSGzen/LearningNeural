import numpy as np

class ReLU():
    def relu_activation(self, input):
        output = []

        _, num_col = np.shape(input)

        for i in range(num_col):
            output.append(max(0, input[0, i]))
        
        return np.array(output)
        
    def relu_derivative(self, input):
        num_row, num_col = np.shape(input)
            
        result = np.empty()
        
        for x_row in range(num_row):
            temp_row = []
            for y_col in range(num_col):
                if result[x_row, y_col] >= 0:
                    temp_row.append(1)
                else:
                    temp_row.append(0)
            temp_row = np.array(temp_row)
            result = np.stack(result, temp_row)
        
        return result
        
