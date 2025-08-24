import numpy as np
from abc import ABC, abstractmethod

class ActivFunc(ABC):
    @abstractmethod
    def calc(self, data): pass

    @abstractmethod
    def calc_deriv(self): pass

class Linear(ActivFunc):
    def calc(self, data):
        return data

    def calc_deriv(self):
        return 1

class Sigmoid(ActivFunc):
    def __init__(self):
        self.result = None

    def calc(self, data):
        self.result = np.where(data >= 0, 1 / (1 + np.exp(-data)), np.exp(data) / (1 + np.exp(data)))
        return self.result

    def calc_deriv(self):
        return self.result * (1 - self.result)

class Tanh(ActivFunc):
    def __init__(self):
        self.result = None

    def calc(self, data):
        self.result = np.tanh(data)
        return self.result

    def calc_deriv(self):
        return 1 - self.result ** 2

class Softmax(ActivFunc):
    def calc(self, data):
        data_exp = np.exp(data - np.max(data, axis=1, keepdims=True))
        return data_exp / np.sum(data_exp, axis=1, keepdims=True)

    # Only with Categorical Cross-entropy loss
    def calc_deriv(self):
        return 1

class ReLU(ActivFunc):
    def __init__(self):
        self.input_data = None

    def calc(self, data):
        self.input_data = data
        return np.maximum(0, data)

    def calc_deriv(self):
        return np.float32(self.input_data > 0)

class LeakyReLU(ActivFunc):
    def __init__(self):
        self.alpha = 0.01
        self.input_data = None

    def calc(self, data):
        self.input_data = data
        return np.where(data > 0, data, data * self.alpha)

    def calc_deriv(self):
        return np.where(self.input_data > 0, np.float32(1), self.alpha)