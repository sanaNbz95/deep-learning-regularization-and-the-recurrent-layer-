import numpy as np
from .Base import BaseLayer


class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.sh, self.sw = stride_shape
        self.pooling_shape_h, self.pooling_shape_w = pooling_shape
        self.input_tensor = None
        self.h_out = None
        self.w_out = None
        self.batch_size = None
        self.channel = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size, self.channel, height, width = self.input_tensor.shape
        # to calculate the result shape
        self.h_out = int(1 + (height - self.pooling_shape_h) / self.sh)
        self.w_out = int(1 + (width - self.pooling_shape_w) / self.sw)

        out = np.zeros((self.batch_size, self.channel, self.h_out, self.w_out))

        for b in range(self.batch_size):
            for ch in range(self.channel):
                for hi in range(self.h_out):
                    for wi in range(self.w_out):
                        out[b, ch, hi, wi] = np.max(
                            input_tensor[b, ch,
                            hi * self.sh: hi * self.sh + self.pooling_shape_h,
                            wi * self.sw: wi * self.sw + self.pooling_shape_w])
        return out

    def backward(self, error_tensor):
        out = np.zeros_like(self.input_tensor)

        for b in range(self.batch_size):
            for ch in range(self.channel):
                for hi in range(self.h_out):
                    for wi in range(self.w_out):
                        i_t, j_t = np.where(np.max(
                            self.input_tensor[b, ch,
                            hi * self.sh: hi * self.sh + self.pooling_shape_h,
                            wi * self.sw: wi * self.sw + self.pooling_shape_w]) == self.input_tensor[
                                                                                                           b,
                                                                                                           ch,
                                                                                                           hi * self.sh: hi * self.sh + self.pooling_shape_h,
                                                                                                           wi * self.sw: wi * self.sw + self.pooling_shape_w])

                        i_t, j_t = i_t[0], j_t[0]

                        out[b, ch, hi * self.sh: hi * self.sh + self.pooling_shape_h,
                        wi * self.sw: wi * self.sw + self.pooling_shape_w][i_t, j_t] += error_tensor[b, ch, hi, wi]

        return out
