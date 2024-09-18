import numpy as np
from layer_class import Layer_Dense
from image_dataset_class import Dataset
from activation_function.activation_func_class import Activation_Function

def test():
    inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

    layer1 = Layer_Dense(4, 5)
    layer2 = Layer_Dense(5, 2)

def main():
    dataset_obj = Dataset()
    train_image_arr = dataset_obj.import_train_image()
    train_label_arr = dataset_obj.import_test_label()
    test_image_arr = dataset_obj.import_test_image()
    test_label_arr = dataset_obj.import_test_label()

main()
    