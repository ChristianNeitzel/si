# Unittest of Evaluation Exercise 4: Implement the Cosine distance function.

import numpy as np
import os
from unittest import TestCase
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv

from si.statistics.cosine_distance import cosine_distance


class TestCosineDistance(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_cosine_distance(self):
        x = np.array([1, 2, 3])
        y = np.array([[1, 0, 0], [0, 1, 0], [1, 2, 3]])

        # Our implementation of cosine distance
        our_distance = cosine_distance(x, y)

        # Using sklearn for validation
        from sklearn.metrics.pairwise import cosine_distances
        sklearn_distance = cosine_distances(x.reshape(1, -1), y).flatten()

        # Check if the distances are close enough
        assert np.allclose(our_distance, sklearn_distance), f"Our distance: {our_distance}, Sklearn distance: {sklearn_distance}"