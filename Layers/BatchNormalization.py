
import numpy as np
from Layers import Base
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()

        self.trainable = True
        self.channels = channels
        self.initialize(None,None)
        # backward
        self._gradient_bias = None
        self._gradient_weights = None
        self._optimizer = None
        self._bias_optimizer = None
        # forward
        self.alpha = 0.8
        self.epsilon = 1e-13
        self.x_hat = None
        self.mean = 0
        self.test_mean = 1
        self.variance = 0
        self.test_variance = 1

    def moving_average(self,mean_axis,var_axis):
        # Training Phase
        new_mean = self.input_tensor.mean(axis=mean_axis)
        new_variance = self.input_tensor.var(axis=var_axis)

        self.test_mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
        self.test_variance = self.alpha * self.variance + (1 - self.alpha) * new_variance

        self.mean = new_mean
        self.variance = new_variance

        return  (self.input_tensor - self.mean) / np.sqrt(self.variance + self.epsilon)

    def forward_2d(self, mean_axis, var_axis):
        if self.testing_phase:

            self.x_hat = (self.input_tensor - self.test_mean) / np.sqrt(self.test_variance + self.epsilon)
        else:
            self.x_hat = self.moving_average(mean_axis,var_axis)

        return self.weights * self.x_hat + self.bias

    def forward_conv_calculator(self, mean_axis, var_axis, channels):
        if self.testing_phase:
            return np.divide(np.subtract(self.input_tensor, self.test_mean.reshape((1, channels, 1, 1))),
                             np.sqrt(self.test_variance.reshape((1, channels, 1, 1)) + self.epsilon))

        new_mean = np.mean(self.input_tensor, axis=mean_axis)
        new_variance = np.var(self.input_tensor, axis=var_axis)

        self.test_mean = np.multiply(self.alpha, self.mean.reshape((1, channels, 1, 1))) + (1 - self.alpha) * new_mean.reshape(
            (1, channels, 1, 1))
        self.test_variance = np.multiply(self.alpha, self.variance.reshape((1, channels, 1, 1))) + (
                1 - self.alpha) * new_variance.reshape((1, channels, 1, 1))

        self.mean = new_mean
        self.variance = new_variance

        return (self.input_tensor - self.mean.reshape((1, channels, 1, 1))) / np.sqrt(
            self.variance.reshape((1, channels, 1, 1)) + self.epsilon)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        is_conv = len(input_tensor.shape) == 4

        mean_axis = 0 if not is_conv else (0, 2, 3)
        var_axis = 0 if not is_conv else (0, 2, 3)
        self.mean = input_tensor.mean(axis=mean_axis)
        self.variance = input_tensor.var(axis=var_axis)

        if not is_conv:
            return self.forward_2d(mean_axis, var_axis)

        batch_size, channels, *_ = self.input_tensor.shape
        self.x_hat = self.forward_conv_calculator(mean_axis, var_axis, channels)

        return self.weights.reshape((1, channels, 1, 1)) * self.x_hat + self.bias.reshape((1, channels, 1, 1))

    def gradient_with_respect_to_bias(self,error_tensor,axis):
         return np.sum(error_tensor, axis=axis)

    def gradient_with_respect_to_weights(self,error_tensor,axis):
         return np.sum(error_tensor * self.x_hat, axis=axis)

    def backward(self, error_tensor):
        is_conv = len(error_tensor.shape) == 4
        axis = 0 if not is_conv else (0,2,3)

        if is_conv:
            wrt_result = compute_bn_gradients(self.reformat(error_tensor), self.reformat(self.input_tensor),
                                       self.weights, self.mean, self.variance, self.epsilon)
            wrt_result = self.reformat(wrt_result)

        else:
            wrt_result = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance,
                                       self.epsilon)

        self.gradient_weights = self.gradient_with_respect_to_weights(error_tensor,axis)
        self.gradient_bias = self.gradient_with_respect_to_bias(error_tensor,axis)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return wrt_result

    def reformat(self, tensor):
        is_conv = len(tensor.shape) == 4
        out = np.zeros_like(tensor)
        if is_conv:
            B, H, M, N = tensor.shape
            out = tensor.reshape((B, H, M * N))
            out = np.transpose(out, (0, 2, 1))
            B, MN, H = out.shape
            out = out.reshape((B * MN, H))
        else:
            B, H, M, N = self.input_tensor.shape
            out = tensor.reshape((B, M * N, H))
            out = np.transpose(out, (0, 2, 1))
            out = out.reshape((B, H, M, N))

        return out

    def initialize(self,_weights_initializer, _bias_initializer):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value

