import os
from unittest import TestCase

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier

# Evaluation Exercise 9: Testing the RandomForestClassifier implementation.
class TestRandomForestClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)
    

    def test_fit(self):
        rf = RandomForestClassifier()
        rf.fit(self.train_dataset)

        self.assertEqual(rf.min_sample_split, 2)
        self.assertEqual(rf.max_depth, 10)
    

    def test_predict(self):
        rf = RandomForestClassifier()
        rf.fit(self.train_dataset)

        predictions = rf.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])


    def test_score(self):
        rf = RandomForestClassifier()
        rf.fit(self.train_dataset)

        accuracy_ = rf.score(self.test_dataset)

        self.assertEqual(round(accuracy_, 2), 1.0)