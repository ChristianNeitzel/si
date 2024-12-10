from abc import ABCMeta, abstractmethod

import numpy as np
from si.base.model import Model


class LossFunction(metaclass=ABCMeta):
    @abstractmethod
    def loss(y_true: np.ndarray, y_pred: np.ndarray):
        raise NotImplementedError
    
    @abstractmethod
    def derivative(L):
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sum((y_true - y_pred) ** 2) / len(y_true)

    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * np.sum(y_true - y_pred) / len(y_true)


class BinaryCrossEntropy(LossFunction):
    def loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)