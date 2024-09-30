from unittest import TestCase
from datasets import DATASETS_PATH

import os
from si.clustering.k_means import KMeans
from si.io.csv_file import read_csv

class TestKMeans(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        from si.data.dataset import Dataset

        k_ = 3
        kmeans = KMeans(k_)
        res = kmeans.fit(self.dataset)
        self.assertEqual(res.centroids.shape[0], 3)
        self.assertEqual(res.labels.shape[0], self.dataset.shape()[0])