from abc import ABCMeta, abstractmethod
import copy

import numpy as np


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError


    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError


    @abstractmethod
    def output_shape(self):
        raise NotImplementedError


    @abstractmethod
    def parameters(self):
        raise NotImplementedError


    def set_input_shape(self, input_shape):
        self._input_shape = input_shape


    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__ # __class__ obtains the class Layer and __name__ obtains the name, these two combined obtains the name of the class.

class DenseLayer(Layer):
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units : int
            The number of units of the layer, AKA the number of neurons, AKA the dimensionality of the output space.
        input_shape : tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None


    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':
        """
        Initialize the weights and biases, and attach an optimizer.

        Parameters
        ----------
        optimizer : object
            Optimizer to use for updating weights and biases.
        """
        n_features = self.input_shape()[0]

        # Initialize weights from a 0 centered uniform distribution [-0.5 and 0.5)
        self.weights = np.random.rand(n_features, self.n_unitest) - 0.5

        # Initialize biases to 0s
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)       # Weights
        self.b_opt = copy.deepcopy(optimizer)       # Biases
        return self


    def parameters(self) -> int:
        """
        Return the total number of parameters (weights + biases).

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output


    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """


        # Computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        # SHAPES: (batch_size, input_columns) = (batch_size, output_columns) * (output_columns, input_columns)
        input_error = np.dot(output_error, self.weights.T)  # output_error * self.weights.T (calculated using np.dot)

        # Computes the weight error: dE/dW = X.T * dE/dY
        # SHAPES: (input_columns, output_columns) = (input_columns, batch_size) * (batch_size, output_columns)
        weights_error = np.dot(self.input.T, output_error)  # self.input.T * output_error (calculated using np.dot)

        # Computes the bias error: dE/dB = dE/dY
        # SHAPES: (1, output_columns) = SUM over the rows of a matrix of shape (batch_size, output_columns)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # Updates parameters
        self.weights = self.w_opt.update(weights_error)
        self.biases = self.b_opt.update(bias_error)

        return input_error


    def output_shape(self):
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units, )
    


