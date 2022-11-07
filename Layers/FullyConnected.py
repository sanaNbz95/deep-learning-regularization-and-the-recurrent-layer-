import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gradient_weights = None
        self.input_tensor = None
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.weights = np.random.random((input_size + 1, output_size))

    def forward(self, input_tensor):
        batch_size, input_size = input_tensor.shape

        last_column = np.ones(batch_size)

        result = np.column_stack((input_tensor, last_column))
        self.input_tensor = result
        y_hat = np.matmul(result, self.weights)
        return y_hat

    def backward(self, error_tensor):
        weight_del = np.delete(self.weights, -1, 0)
        prev_error_tensor = np.matmul(error_tensor, weight_del.transpose())
        gradient = np.matmul(self.input_tensor.transpose(), error_tensor)
        self.gradient_weights = gradient
        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, gradient)
        return prev_error_tensor

    # exercise 2
    def initialize(self, weights_initializer, bias_initializer):
        # row wise
        self.weights = np.vstack((weights_initializer.initialize((self.input_size, self.output_size), self.input_size,
                                                                 self.output_size),
                                  bias_initializer.initialize((1, self.output_size), 1, self.output_size)))

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
