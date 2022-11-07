
from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .TanH import TanH
from .Sigmoid import Sigmoid

import copy
import numpy as np


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.regularization_loss = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)

        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.hidden_fc_layer = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.output_fc_layer = FullyConnected(self.hidden_size, self.output_size)

        self.hidden_fc_layers_input_tensors = np.ndarray([])
        self.hidden_fc_layer_gradient_weights = []

        self.output_fc_layer_gradient_weights = []
        self.output_fc_input_tensors = []

        self.sigmoid_activations = []
        self.tanh_activations = []

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc_layer.initialize(weights_initializer, bias_initializer)
        self.output_fc_layer.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.sigmoid_activations = []
        self.tanh_activations = []
        self.output_fc_input_tensors = []
        self.hidden_fc_layers_input_tensors = []
        prev_hidden_state = self.hidden_state if self.memorize else np.zeros(self.hidden_size)
        batch_size,*_ = input_tensor.shape
        output_tensor = np.ndarray((batch_size,self.output_size))

        for i in range(input_tensor.shape[0]):

            x_tilda = np.concatenate([prev_hidden_state, input_tensor[i]]).reshape(1, -1)


            tan_input_tensor = self.hidden_fc_layer.forward(x_tilda)
            current_hidden_state = self.tanh.forward(tan_input_tensor)

            prev_hidden_state = current_hidden_state[0]


            transition_of_hy = self.output_fc_layer.forward(current_hidden_state)
            sigmoid_output_tensor = self.sigmoid.forward(transition_of_hy)

            output_tensor[i]=sigmoid_output_tensor[0]

            self.hidden_fc_layers_input_tensors.append(self.hidden_fc_layer.input_tensor)
            self.output_fc_input_tensors.append(self.output_fc_layer.input_tensor)
            self.sigmoid_activations.append(self.sigmoid.activations)
            self.tanh_activations.append(self.tanh.activations)

        self.hidden_state = current_hidden_state[0]

        return output_tensor

    def backward(self, error_tensor):
        self.gradient_weights = np.zeros_like(self.hidden_fc_layer.weights)
        self.output_fc_layer_gradient_weights = np.zeros_like(self.output_fc_layer.weights)
        gradient_prev_hidden_state = 0
        batch_size,*_ = error_tensor.shape
        gradient_inputs = np.zeros((batch_size,self.input_size))

        time_step = error_tensor.shape[0] - 1

        while time_step >= 0:
            self.sigmoid.activations = self.sigmoid_activations[time_step]
            sigmoid_error = self.sigmoid.backward(error_tensor[time_step])

            self.output_fc_layer.input_tensor = self.output_fc_input_tensors[time_step]
            output_fc_layer_error = self.output_fc_layer.backward(sigmoid_error)

            self.tanh.activations = self.tanh_activations[time_step]
            tanh_error = self.tanh.backward(output_fc_layer_error + gradient_prev_hidden_state)

            self.hidden_fc_layer.input_tensor = self.hidden_fc_layers_input_tensors[time_step]
            hidden_fc_layer_error = self.hidden_fc_layer.backward(tanh_error)

            gradient_prev_hidden_state = hidden_fc_layer_error[:, :self.hidden_size]
            gradient_with_respect_to_input = hidden_fc_layer_error[:, self.hidden_size:]
            gradient_inputs[time_step]=gradient_with_respect_to_input[0]

            self.gradient_weights+=self.hidden_fc_layer.gradient_weights
            self.output_fc_layer_gradient_weights+=self.output_fc_layer.gradient_weights

            time_step -= 1


        if self.optimizer:
            self.output_fc_layer.weights = self.optimizer.calculate_update(self.output_fc_layer.weights,
                                                                           self.output_fc_layer_gradient_weights)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return gradient_inputs

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.hidden_fc_layer.weights

    @weights.setter
    def weights(self, value):
        self.hidden_fc_layer.weights = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def calculate_regularization_loss(self):
        if self.optimizer.regularizer:
            self.regularization_loss += self.optimizer.regularizer.norm(self.hidden_fc_layer.weights)
            self.regularization_loss += self.optimizer.regularizer.norm(self.output_fc_layer.weights)
        return self.regularization_loss