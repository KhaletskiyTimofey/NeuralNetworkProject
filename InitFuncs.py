import numpy as np
from abc import ABC, abstractmethod

class InitFunc(ABC):
    @abstractmethod
    def init_weights(self, input_size, output_size): pass

class Xavier(InitFunc):
    def init_weights(self, input_size, output_size):
        return np.random.normal(0, np.sqrt(2 / (input_size + output_size)), (input_size, output_size))

class He(InitFunc):
    def init_weights(self, input_size, output_size):
        return np.random.normal(0, np.sqrt(2 / input_size), (input_size, output_size))