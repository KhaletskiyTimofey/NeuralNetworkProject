import numpy as np
from abc import ABC, abstractmethod

class LossFunc(ABC):
    @abstractmethod
    def calc(self): pass

    @abstractmethod
    def calc_deriv(self, output_data, desired_output): pass

class MSE(LossFunc):
    def __init__(self):
        self.output_data = None
        self.desired_output = None

    def calc(self):
        return np.mean((self.output_data - self.desired_output) ** 2)

    def calc_deriv(self, output_data, desired_output):
        self.output_data = output_data
        self.desired_output = desired_output
        return (output_data - desired_output) * 2

class BinaryCrossEntropy(LossFunc):
    bias = 1e-15

    def __init__(self):
        self.output_data = None
        self.desired_output = None

    def calc(self):
        return -np.mean(self.desired_output * np.log(self.output_data + self.bias) +
                        (1 - self.desired_output) * np.log(1 - self.output_data + self.bias))

    # Only with Sigmoid activation
    def calc_deriv(self, output_data, desired_output):
        self.output_data = output_data
        self.desired_output = desired_output
        return output_data - desired_output

class CategoricalCrossEntropy(LossFunc):
    bias = 1e-15

    def __init__(self):
        self.output_data = None
        self.desired_output = None

    def calc(self):
        return -np.mean(np.sum(self.desired_output * np.log(self.output_data + self.bias), axis=1))

    # Only with Softmax activation
    def calc_deriv(self, output_data, desired_output):
        self.output_data = output_data
        self.desired_output = desired_output
        return output_data - desired_output