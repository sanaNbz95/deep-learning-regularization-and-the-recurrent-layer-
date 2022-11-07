import numpy as np
from .Base import BaseLayer


class SoftMax(BaseLayer):

    def __init__(self):
        super().__init__()
        self.y_hat = None

    def forward(self, input_tensor):
        max_x = np.amax(input_tensor, axis=1, keepdims=True)
        shifted_tensor = input_tensor - max_x
        a = np.exp(shifted_tensor)
        sum_exp = np.sum(np.exp(shifted_tensor), axis=1, keepdims=True)

        self.y_hat = a / sum_exp
        return self.y_hat

    def backward(self, error_tensor):
        sigma_y_hat_error = np.sum(np.multiply(error_tensor, self.y_hat), axis=1, keepdims=True)
        error_tensor = error_tensor - sigma_y_hat_error
        return np.multiply(self.y_hat, error_tensor)
