import math
import numpy as np


#     exercise 3
class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = float(learning_rate)

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        try:
            return updated_weight - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        except:
            return updated_weight


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.momentum_rate * self.prev_v - self.learning_rate * gradient_tensor
        self.prev_v = v
        updated_weight = weight_tensor + v

        try:
            return updated_weight - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        except:
          return updated_weight


class Adam(Optimizer):
    def __init__(self, learning_rate, mo, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mo = mo
        self.rho = rho
        self.prev_v = 0
        self.prev_r = 0
        self.iter = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = self.mo * self.prev_v + (1. - self.mo) * gradient_tensor
        self.prev_v = v

        r = self.rho * self.prev_r + (1. - self.rho) * np.power(gradient_tensor, 2)
        self.prev_r = r

        v_hat = v / (1. - np.power(self.mo, self.iter))
        r_hat = r / (1. - np.power(self.rho, self.iter))

        self.iter += 1

        updated_weight = weight_tensor - (self.learning_rate * (v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)))

        try:
            return updated_weight - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        except:
            return updated_weight
