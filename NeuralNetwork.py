import copy
import numpy as np

from Layers.SoftMax import SoftMax


class NeuralNetwork:

    def __init__(self, optimizer,weights_initializer, bias_initializer):
        self.label_tensor = None
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None

    # exercise 3
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        regularization_loss =0
        for layer in self.layers:

            input_tensor = layer.forward(input_tensor)
            # exercise 3
            try:
                regularization_loss+= self.optimizer.regularizer.norm(layer.weights)
            except:
                pass
            layer.testing_phase = True
        return self.loss_layer.forward(input_tensor+regularization_loss, self.label_tensor)

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            # exercise 2
            layer.initialize(self.weights_initializer,self.bias_initializer)

        self.layers.append(layer)

    def train(self, iteration):
        self.phase= True


        for _ in np.arange(iteration):

            self.loss.append(self.forward())
            self.backward()



    def test(self, input_tensor):
        self.phase= False
        soft_max=SoftMax()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)



        return soft_max.forward(input_tensor)


