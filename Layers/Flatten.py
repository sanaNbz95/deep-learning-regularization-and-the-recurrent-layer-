import numpy as np
from .Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.batch_size = None

    def forward(self, input_tensor):
        self.batch_size, *self.input_shape = input_tensor.shape
        self.input_shape = tuple(self.input_shape)
        return np.reshape(input_tensor, (self.batch_size, np.prod(self.input_shape)))

    def backward(self, error_tensor):

        return np.reshape(error_tensor, (self.batch_size, *self.input_shape))

# from Layers import Base


# class Flatten(Base.BaseLayer):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input_tensor):
#         self.shape = input_tensor.shape
#         if len(input_tensor.shape) == 2:
#             batch_size, _ = self.shape
#         elif len(input_tensor.shape) == 4:
#             batch_size, _, _, _ = self.shape
#         return input_tensor.reshape(batch_size, -1)
#
#     def backward(self, error_tensor):
#         return error_tensor.reshape(self.shape)