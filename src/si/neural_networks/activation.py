from abc import abstractmethod
from typing import Union

import numpy as np

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """
    Base class for activation layers.
    """
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
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error: float) -> Union[float, np.ndarray]:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output error of the layer.
        """
        return self.derivative(self.input) * output_error
    
    @abstractmethod
    def activation_function(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The output of the layer.
        """
        raise NotImplementedError
    
    @abstractmethod
    def derivative(self, input: np.ndarray) -> Union[float, np.ndarray]:
        """
        Derivative of the activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        Union[float, numpy.ndarray]
            The derivative of the activation function.
        """
        raise NotImplementedError
    
    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape
    
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0
    
class SigmoidActivation(ActivationLayer):
    """
    Sigmoid activation function.
    """
    def activation_function(self, input: np.ndarray):
        """
        Sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return 1 / (1 + np.exp(-input))

    def derivative(self, input: np.ndarray):
        """
        Derivative of the sigmoid activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return self.activation_function(input) * (1 - self.activation_function(input))
    
    
class ReLUActivation(ActivationLayer):
    """
    ReLU activation function.
    """
    def activation_function(self, input: np.ndarray):
        """
        ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        return np.maximum(0, input)

    def derivative(self, input: np.ndarray):
        """
        Derivative of the ReLU activation function.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        return np.where(input >= 0, 1, 0)


# Evaluation Exercise 13.1: TanhActivation class implementation.
class TanhActivation(ActivationLayer):
    """
    Hyperbolic Tangent (Tanh) activation function.
    """
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Tanh activation function.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output of the tanh activation function.
        """
        # np.cosh(input) = 1/2 * (np.exp(input) + np.exp(-input))
        # np.sinh(input) = 1/2 * (np.exp(input) - np.exp(-input))
        # np.tanh(input) = np.sinh(input) / np.cosh(input)
        # np.tanh(input) = (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        
        exp_pos = np.exp(input) # e^x
        exp_neg = np.exp(-input) # e^(-x)
        return (exp_pos - exp_neg) / (exp_pos + exp_neg) # Should correspond to np.tanh(input)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the tanh activation function.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the tanh activation function.
        """
        # Note: x -> input; f(x) -> self.activation_function(input)

        return 1 - (self.activation_function(input) ** 2) # Should correspond to: 1 - np.tanh(input) ** 2


# Evaluation Exercise 13.2: SoftmaxActivation class implementation.
class SoftmaxActivation(ActivationLayer):
    """
    Softmax activation function.
    """
    def activation_function(self, input: np.ndarray) -> np.ndarray:
        """
        Softmax activation function.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The output probability for each class.
        """
        # Numerical stability fix to prevent large exponents
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def derivative(self, input: np.ndarray) -> np.ndarray:
        """
        Derivative of the softmax activation function.

        Parameters
        ----------
        input : numpy.ndarray
            The input to the layer.

        Returns
        -------
        numpy.ndarray
            The derivative of the activation function.
        """
        # Note: x -> input; f(x) -> self.activation_function(input)
        return self.activation_function(input) * (1 - self.activation_function(input))