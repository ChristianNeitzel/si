from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH

import os

from si.io.csv_file import read_csv
from si.metrics.rmse import rmse
from si.models.knn_classifier import KNNClassifier
from si.models.knn_regressor import KNNRegressor
from si.model_selection.split import train_test_split

class TestKNN(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNClassifier(k=3)

        knn.fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNClassifier(k=1)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertTrue(np.all(predictions == test_dataset.y))

    def test_score(self):
        knn = KNNClassifier(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        self.assertEqual(score, 1)


# Evaluation Exercise 7.3: Test the "KNNRegressor" class using the "cpu.csv" dataset (regression)
class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)

        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))

    def test_score(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)

        score = knn.score(test_dataset)
        expected_score = rmse(test_dataset.y, predictions) # rmse(dataset.y, predictions)

        self.assertAlmostEqual(score, expected_score)