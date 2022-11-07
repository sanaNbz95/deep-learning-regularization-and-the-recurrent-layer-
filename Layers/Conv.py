from .Base import BaseLayer

import numpy as np
from scipy import signal


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = (stride_shape[0], stride_shape[0]) if len(stride_shape) == 1 else stride_shape
        # [channel,m] [channel,m,n]
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # gradients with respect to the weights
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.rand(num_kernels)
        # gradient with respect to the weights and bias in backward pass
        self._gradient_weights = np.zeros(self.weights.shape)
        self._gradient_bias = None

        self._optimizer = None
        self._bias_optimizer = None

        self.is_2d = len(convolution_shape) == 3

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(
            (self.num_kernels, *self.convolution_shape),
            np.prod(list(self.convolution_shape)),
            np.prod([self.num_kernels, *self.convolution_shape[1:]])
        )

        self.bias = bias_initializer.initialize(
            self.bias.shape,
            1,
            self.num_kernels
        )

    def forward(self, input_tensor):
        # [b,c,y,x] : input_tensor
        self.input_tensor = input_tensor

        batch_size, channels, y, x = input_tensor.shape if self.is_2d else (*input_tensor.shape, None)
        sh, sw = self.stride_shape

        output_shape = [int(np.ceil(y / sh))]


        if self.is_2d:
            output_shape.append(int(np.ceil(x / sw)))

        result = np.zeros((batch_size, self.num_kernels, *output_shape))

        for b in range(batch_size):
            for kernel in range(self.num_kernels):
                out = []
                for ch in range(channels):
                    out.append(
                        # Cross-correlate two N-dimensional arrays
                        signal.correlate(input_tensor[b, ch], self.weights[kernel, ch], mode='same', method='direct')
                    )
                out = np.array(out).sum(axis=0)

                out = out[::sh, ::sw] if self.is_2d else out[::sh]
                result[b, kernel] = out + self.bias[kernel]

        return result

    def calc_gradient_with_respect_to_bias(self, error_tensor):
        gradient_bias_axis = (0, 2, 3) if self.is_2d else (0, 2)

        self._gradient_bias = np.sum(error_tensor, axis=gradient_bias_axis)

    def calc_gradient_with_respect_to_weights(self, error_tensor):
        self._gradient_weights = np.zeros(self.weights.shape)

        batch_size, channels, y, x = self.input_tensor.shape if self.is_2d else (*self.input_tensor.shape, None)
        sh, sw = self.stride_shape
        *_, conv_shape_x, conv_shape_y = self.convolution_shape

        for b in range(batch_size):
            for ch in range(channels):
                for kernel in range(self.num_kernels):
                    # horizontal padding(right, left) (f - 1) /2
                    h_padding = (int(np.ceil((conv_shape_x - 1) / 2)), int(np.floor((conv_shape_x - 1) / 2)))

                    if self.is_2d:
                        # vertical padding(up, down)(f - 1) /2
                        v_padding = (int(np.ceil((conv_shape_y - 1) / 2)), int(np.floor((conv_shape_y - 1) / 2)))

                        upsample_arr = np.zeros((y, x))
                        upsample_arr[::sh, ::sw] = error_tensor[b, kernel]

                        padded_input_tensor = np.pad(self.input_tensor[b, ch],
                                                     [h_padding, v_padding])
                    else:
                        upsample_arr = np.zeros(y)
                        upsample_arr[::sh] = error_tensor[b, kernel]

                        padded_input_tensor = np.pad(self.input_tensor[b, ch], [h_padding])

                    self.gradient_weights[kernel, ch] += signal.correlate(
                        padded_input_tensor, upsample_arr, mode='valid'
                    )

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)

    def calc_gradient_with_respect_to_lower_layers(self, error_tensor):
        sh, sw = self.stride_shape

        input_gradient = np.zeros(self.input_tensor.shape)
        w_copy = self.weights.copy()
        # weight matrix H,S,N,M, convolve with S,H,N,M
        transpose_axis = (1, 0, 2, 3) if self.is_2d else (1, 0, 2)
        w_copy = np.transpose(w_copy, transpose_axis)

        batch_size, *_ = error_tensor.shape
        w_kernel, w_channels, *_ = w_copy.shape
        # [b,c,y,x] : input_tensor
        *_, y, x = self.input_tensor.shape if self.is_2d else (*self.input_tensor.shape, None)

        for b in range(batch_size):
            for kernel in range(w_kernel):
                convolutions = []

                for ch in range(w_channels):
                    if self.is_2d:
                        upsample_arr = np.zeros((y, x))
                        upsample_arr[::sh, ::sw] = error_tensor[b, ch]
                    else:
                        upsample_arr = np.zeros(y)
                        upsample_arr[::sh] = error_tensor[b, kernel]

                    convolved_upsample = signal.convolve(upsample_arr, w_copy[kernel, ch], mode='same', method='direct')
                    convolutions.append(convolved_upsample)

                convolutions = np.array(convolutions).sum(axis=0)

                input_gradient[b, kernel] = convolutions
        return input_gradient

    def backward(self, error_tensor):
        self.calc_gradient_with_respect_to_bias(error_tensor)

        self.calc_gradient_with_respect_to_weights(error_tensor)

        return self.calc_gradient_with_respect_to_lower_layers(error_tensor)

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
