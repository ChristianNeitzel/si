# Evaluation Exercise 8: Testing the Lasso Regression implementation.

from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.lasso_regression import LassoRegression

class TestLassoRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        ridge = LassoRegression()
        ridge.fit(self.train_dataset)

    def test_predict(self):
        ridge = LassoRegression()
        ridge.fit(self.train_dataset)
        predictions = ridge.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    
    def test_score(self):
        ridge = LassoRegression(scale=True, patience=5)
        ridge.fit(self.train_dataset)
        mse_ = ridge.score(self.test_dataset)

        self.assertEqual(round(mse_, 2), 5777.56)