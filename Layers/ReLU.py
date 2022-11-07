import numpy as np
from .Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(np.maximum(input_tensor, 0.0), input_tensor, 0)

    def backward(self, error_tensor):
        return np.where(self.input_tensor <= 0, 0, error_tensor)
