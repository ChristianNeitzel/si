from abc import abstractmethod, ABCMeta
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

        # Initialize weights randomly between -0.5 and 0.5
        #self.weights = np.random.uniform(-0.5, 0.5, size=(n_features, self.n_units))
        self.weights = np.random.rand(n_features, self.n_unitest) - 0.5

        # Initialize biases to 0s
        self.biases = np.zeros((1, self.n_units))

        # Optimizer setup
        self.optimizer_weights = copy.deepcopy(optimizer)
        self.optimizer_biases = copy.deepcopy(optimizer)
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


    def backward_propagation(self, error, learning_rate):
        """
        Perform backward propagation through the layer.

        Parameters
        ----------
        error : numpy.ndarray
            Gradient of the loss with respect to the output of the layer.
        learning_rate : float
            Learning rate for the optimizer.
        
        Returns
        -------
        numpy.ndarray
            Gradient of the loss with respect to the input of the layer.
        """
        # Calculate gradients
        weights_gradient = np.dot(self.input.T, error)
        biases_gradient = np.sum(error, axis=0, keepdims=True)
        input_error = np.dot(error, self.weights.T)

        # Update parameters using the optimizer
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_error


    def output_shape(self):
        """
        Return the shape of the layer's output.

        Returns
        -------
        tuple
            Shape of the output (n_units, ).
        """
        return (self.n_units, )
