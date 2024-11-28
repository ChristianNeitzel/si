import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression

# Evaluation Exercise 11.2: Test the randomized_search_cv function implementation.
class TestRandomizedSearch(TestCase):
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search_cross_validation(self):

        model = LogisticRegression()

        # Parameter grid
        parameter_grid_ = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200)
        }

        # Cross validate the model
        results_ = randomized_search_cv(model,
                                        self.dataset,
                                        hyperparameter_grid=parameter_grid_,
                                        cv=3,
                                        n_iter=10)

        self.assertEqual(len(results_["scores"]), 10)

        best_hyperparameters = results_['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3)

        best_score = results_['best_score']
        self.assertEqual(np.round(best_score, 2), 0.97)