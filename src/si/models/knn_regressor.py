# Evaluation Exercise 7.2: Implementing KNNRegressor.

from typing import Callable, Union

import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.rmse import rmse


class KNNRegressor(Model):
    """
    K-Nearest Neighbors (KNN) Regressor.

    Predicts the value of a new sample by averaging the values of its k-nearest neighbors in the training data.
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
        self.dataset = None

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
        self.dataset = dataset
        return self
    
    def _get_closest_value(self, sample: np.ndarray) -> Union[int, float]:
        """
        It returns the closest value of the given sample.
        (modified version of _get_closest_label() from knn_classifier.py so it obtains the average value of the k nearest neighbors)

        Parameters
        ----------
        sample : numpy.ndarray
            The sample to get the closest value of.

        Returns
        -------
        value : int or float
            The closest value.
        """
        distances = self.distance(sample, self.dataset.X)                   # Compute the distance between the sample and the dataset
        k_nearest_neighbors = np.argsort(distances)[:self.k]                # Get the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]    # Get the labels of the k nearest neighbors
        value = np.sum(k_nearest_neighbors_labels) / self.k                 # Get the average value of the k nearest neighbors

        return value
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the values for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict the target values for.

        Returns
        -------
        predictions : numpy.ndarray
            The predicted target values.
        """
        predictions = np.apply_along_axis(self._get_closest_value, axis=1, arr=dataset.X)
        return predictions


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the RMSE score for the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on.

        predictions: numpy.ndarray
            An array with the predictions.

        Returns
        -------
        float
            The RMSE value.
        """
        return rmse(dataset.y, predictions)
    
# (Evaluation Exercise 7.3 present in file "si/tests/unit_tests/test_knn.py")