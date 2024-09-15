import os
import idx2numpy
import numpy as np

current_path = os.getcwd()

class Dataset:
    def import_train_image(self):
        file_address = r'\uncompressed_dataset\train-images.idx3-ubyte'
        full_address = current_path + file_address
        print(full_address)
        train_images_arr = idx2numpy.convert_from_file(full_address)

        return train_images_arr
    
    def import_train_label(self):
        file_address = r"\uncompressed_dataset\train-labels.idx1-ubyte"
        full_address = current_path + file_address
        train_labels_arr = idx2numpy.convert_from_file(full_address)

        return train_labels_arr

    def import_test_image(self):
        file_address = r"\uncompressed_dataset\t10k-images.idx3-ubyte"
        full_address = current_path + file_address
        test_images_arr = idx2numpy.convert_from_file(full_address)

        return test_images_arr
    
    def import_test_label(self):
        file_address = r"\uncompressed_dataset\t10k-labels.idx1-ubyte"
        full_address = current_path + file_address
        test_labels_arr = idx2numpy.convert_from_file(full_address)

        return test_labels_arr
