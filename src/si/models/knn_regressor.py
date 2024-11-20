from typing import Callable

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse

# Evaluation Exercise 7.2: Implementing KNNRegressor.
class KNNRegressor(Model):
    """
    KNN Regressor
    The k-Nearest Neighbors regressor predicts the value of a new sample by averaging
    the values of its k-nearest neighbors in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use.
    distance: Callable
        The distance function to use.

    Attributes
    ----------
    train_dataset: Dataset
        The training dataset.
    """
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN regressor.

        Parameters
        ----------
        k : int
            The number of nearest neighbors to use.
        distance: Callable
            The distance function to use.
        """
        # parameters
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance

        # attributes
        self.train_dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Store the training dataset.

        Parameters
        ----------
        dataset : Dataset
            The training dataset

        Returns
        -------
        self : KNNRegressor
            The fitted model.
        """
        self.train_dataset = dataset
        return self
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the values for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict the target values for.

        Returns
        -------
        np.ndarray
            The predicted target values.
        """
        predictions = []

        for sample in dataset.X:
            # Calculate distances to all training samples
            distances = self.distance(sample, self.train_dataset.X)
            
            # Get indices of the k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Average the target values of the k nearest neighbors
            k_neighbors = self.train_dataset.y[k_indices]
            predictions.append(np.mean(k_neighbors))
        
        return np.array(predictions)


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the RMSE score for the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on.

        predictions: np.ndarray
            An array with the predictions.

        Returns
        -------
        float
            The RMSE value.
        """
        return rmse(dataset.y, predictions)