import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    It returns the mean squared error for the y_pred variable.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    mse: float
        The mean squared error of the model
    """

    # Calculate the Mean Squared Error (MSE)
    return np.sum((y_true - y_pred) ** 2) / len(y_true)   # Using numpy's sum() function (np.sum()), we won't have to specify the need to iterate the indexes of y_true or y_pred

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    It returns the derivative of the mean squared error for the y_pred variable.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    mse_derivative: np.ndarray
        The derivative of the mean squared error
    """
    return 2 * np.sum(y_true - y_pred) / len(y_true)