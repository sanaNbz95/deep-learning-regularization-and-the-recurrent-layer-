import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.pre = None

    def forward(self, prediction_tensor, label_tensor):
        self.pre = prediction_tensor
        return np.sum(np.where(label_tensor == 1, - np.log(self.pre+ np.finfo(prediction_tensor.dtype).eps), 0))

    def backward(self, label_tensor):
        return -1 * np.divide(label_tensor, self.pre )
