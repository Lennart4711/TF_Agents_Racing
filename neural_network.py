import numpy as np

# sigmoid function - returns number between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
Neural network with multiple layers.
"""


class NeuralNetwork:
    def __init__(self, shape):
        ### SHAPE ###
        self.size = len(shape)  # number of layers
        self.shape = shape  # shape of nn
        self.weights = []  # weights

        ### ZEROS ###
        self.weights.extend(
            np.zeros((self.shape[i - 1], self.shape[i])) for i in range(1, self.size)
        )

    # set random weights
    def set_random_weights(self):
        for i in range(1, self.size):
            self.weights[i - 1] = np.random.rand(self.shape[i - 1], self.shape[i]) - 0.5

    # set weights
    def set_weights(self, weights):
        for count, layer in enumerate(weights):
            self.weights[count] = layer

    # output
    def forward(self, inp):
        # inp array shape (5,)
        layer = inp
        for i in range(self.size - 1):
            layer = np.dot(layer, self.weights[i])
        # out array shape (2,)
        for ind in range(len(layer)):
            layer[ind] = sigmoid(layer[ind])
        return layer

    # returns copy of itself
    def get_deep_copy(self):
        n = NeuralNetwork(self.shape)
        for i in range(self.size - 1):
            n.weights[i] = self.weights[i]
        return n

    # returns slightly mutated version of itself
    def reproduce(self, mutation):
        n = NeuralNetwork(self.shape)
        for i in range(1, self.size):
            n.weights[i - 1] = self.weights[i - 1] + (
                (np.random.rand(self.shape[i - 1], self.shape[i]) - 0.5) * mutation
            )
        return n
