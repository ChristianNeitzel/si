# Evaluation Exercise 7.1: Implementing RMSE (Root Mean Square Deviation).

import numpy as np


def rmse(y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : numpy.ndarray
        Array of true values.
    y_pred : numpy.ndarray
        Array of predicted values.

    Returns
    -------
    float
        The RMSE value.
    """
    # # Iterating through the indices
    # # Without numpy
    # n = len(y_true)
    # total = 0
    # for i in range(n):
    #     total += (Y_pred[i] - y_true[i]) ** 2
    # mse = total / n
    # rmse = np.sqrt(mse)

    # # With numpy
    # mse = np.sum((y_true - Y_pred) ** 2) / len(y_true)
    # rmse = np.sqrt(mse)

    return np.sqrt(np.sum((y_true - Y_pred) ** 2) / len(y_true))