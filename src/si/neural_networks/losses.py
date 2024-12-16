from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)
        # return np.sum((y_true - y_pred) ** 2) / len(y_true)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true
        return 2 * (y_pred - y_true) / y_true.size
        # return 2 * np.sum(y_true - y_pred) / len(y_true)


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        # return -sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)
        # return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
    

# Evaluation Exercise 14: CategoricalCrossEntropy class implementation
class CategoricalCrossEntropy(LossFunction):
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes the categorical cross-entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            True labels (one-hot encoded).
        y_pred : numpy.ndarray
            Predicted labels.

        Returns
        -------
        float
            The categorical cross-entropy loss.
        """
        # Clip predictions to avoid log(0) and numerical instability
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]   # Compute and output the categorical cross-entropy

    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the categorical cross-entropy loss.

        Parameters
        ----------
        y_true : numpy.ndarray
            True labels (one-hot encoded).
        y_pred : numpy.ndarray
            Predicted probabilities.

        Returns
        -------
        numpy.ndarray
            The gradient of the loss with respect to the predictions.
        """
        # Clip predictions to avoid division by zero and numerical instability
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        return -(y_true / y_pred)