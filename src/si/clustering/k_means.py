from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.base.model import Model
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance

class KMeans(Transformer, Model):
    """
    It performs k-means clustering on the dataset.
    It groups samples into k clusters by trying to minimize the distance between samples and their closest centroid.
    It returns the centroids and the indexes of the closest centroid for each point.

    Parameters
    ----------
    k: int
        Number of clusters.
    max_iter: int
        Maximum number of iterations.
    distance: Callable
        Distance function.

    Attributes
    ----------
    centroids: np.array
        Centroids of the clusters.
    labels: np.array
        Labels of the clusters.
    """

    def __init__(self, k: int, max_iter: int = 1000, distance: Callable = euclidean_distance, **kwargs):
        """
        K-means clustering algorithm.

        Parameters
        ----------
        k: int
            Number of clusters.
        max_iter: int
            Maximum number of iterations.
        distance: Callable
            Distance function.

        Note: max_iter by default set to 1000; distance by default follows euclidean_distance() function
        """
        # parameters
        super().__init__(**kwargs)
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        # attributes
        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        """
        It generates initial k centroids.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.
        """
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]  # Seeds are the indexes of each point
        self.centroids = dataset.X[seeds]                           # Obtain the centroids

    def _get_closest_centroid(self, sample: np.ndarray) -> np.ndarray:
        """
        Get the closest centroid to each data point.

        Parameters
        ----------
        sample : np.ndarray, shape=(n_features,)
            A sample.

        Returns
        -------
        np.ndarray
            The closest centroid to each data point.
        """
        centroids_distances = self.distance(sample, self.centroids)
        closest_centroid_index = np.argmin(centroids_distances, axis=0)
        return closest_centroid_index

    def _fit(self, dataset: Dataset) -> 'KMeans':
        """
        It fits k-means clustering on the dataset.
        The k-means algorithm initializes the centroids and then iteratively updates them until convergence or max_iter.
        Convergence is reached when the centroids do not change anymore.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        KMeans
            KMeans object.
        """
        # Generate initial centroids
        self._init_centroids(dataset=dataset)

        # Fitting the K-Means
        convergence = False
        i = 0
        labels = np.zeros(dataset.shape()[0])
        while not convergence and i < self.max_iter:

            # Get closest centroid
            new_labels = np.apply_along_axis(self._get_closest_centroid, axis=1, arr=dataset.X)

            # Compute the new centroids
            centroids = []
            for j in range(self.k):
                centroid = np.mean(dataset.X[new_labels==j], axis=0)
                centroids.append(centroid)
            
            self.centroids = np.array(centroids)

            # Check if the centroids have changed
            convergence = np.all(new_labels == self.labels)     # Alternatively, writting "convergence = not np.any(new_labels != labels)" also works
            
            # Replace labels
            labels = new_labels

            # Increment counting
            i += 1

        self.labels = labels
        return self

    def _transform(self, dataset: Dataset) -> np.ndarray:
        """
        It transforms the dataset.
        It computes the distance between each sample and the closest centroid.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Transformed dataset.
        """
        pass

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the labels of the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset object.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        pass


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset_ = Dataset.from_random(100, 5)

    k_ = 3
    kmeans = KMeans(k_)
    res = kmeans.fit_transform(dataset_)
    predictions = kmeans.predict(dataset_)
    print(res.shape)
    print(predictions.shape)
