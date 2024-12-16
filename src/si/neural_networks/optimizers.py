from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):
    def __init__(self, learning_rate: float):
        """
        Initializes the Optimizer class.
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w : numpy.ndarray
            The current weights of the layer.
        grad_loss_w : numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD).
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initializes the optimizer.

        Parameters
        ----------
        learning_rate : float
            The learning rate to use for updating the weights.
        momentum : float
            The momentum to use for updating the weights.
        """
        # Arguments
        super().__init__(learning_rate) # Alternatively, replace learning_rate in both the super() and the __init() parameters with **kwargs.
        self.momentum = momentum

        # Parameters
        self.retained_gradient = None   # Estimated parameters are set to None in the constructor, as they will be obtained later.

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w : numpy.ndarray
            The current weights of the layer.
        grad_loss_w : numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        # Verify if retained_gradient is initialized
        if self.retained_gradient is None:                  # If retained_gradient was not initialized:
            self.retained_gradient = np.zeros(np.shape(w))  # Initialize retained_gradient as a matrix of zeroes

        # If retained_gradient was initialized:
        # Compute and update the retained gradient
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient  # Output the updated weights


# Evaluation Exercise 15: Adam class implementation.
class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam).
    """
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer class.

        Parameters
        ----------
        learning_rate : float
            The learning rate for updating the weights.
        beta_1 : float
            The exponential decay rate for the 1st moment estimates.
        beta_2 : float
            The exponential decay rate for the 2nd moment estimates.
        epsilon : float
            A small constant for numerical stability.
        """
        # Arguments
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Parameters
        self.m = None   # Moving average m (First moment vector)
        self.v = None   # Moving average v (Second moment vector)
        self.t = 0      # Time step

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w : numpy.ndarray
            The current weights of the layer.
        grad_loss_w : numpy.ndarray
            The gradient of the loss function with respect to the weights.
        
        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        # If m and v are not initialized -> initialize them as matrices of zeros.
        if self.m is None or self.v is None:      
            self.m = np.zeros(np.shape(w)) # Maybe try if np.zeros_like(w) instead of np.zeros(np.shape(w)) also works?
            self.v = np.zeros(np.shape(w))

        # Update t
        self.t += 1

        # If m and v are initialized:
        # Compute and update m (Update biased first moment estimate)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

        # Compute and update v (Update biased second raw moment estimate)
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        # Compute m_hat (Compute bias-corrected first moment estimate)
        m_hat = self.m / (1 - self.beta_1 ** self.t)

        # Compute v_hat (Compute bias-corrected second moment estimate)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        return w - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon)) # Output the updated weights
