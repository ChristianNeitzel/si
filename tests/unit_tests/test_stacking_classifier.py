import os
from unittest import TestCase

from datasets import DATASETS_PATH
from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

# Evaluation Exercise 10.2: Testing the StackingClassifier ensemble implementation.
class TestStackingClassifier(TestCase):
    
    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

        # Base models
        self.knn = KNNClassifier()
        self.lg = LogisticRegression()
        self.dt = DecisionTreeClassifier()

        # Final model
        self.knn_final = KNNClassifier()

        # Stacking model
        self.stacking = StackingClassifier(
            models=[self.knn, self.lg, self.dt], 
            final_model=self.knn_final
            )
        

    def test_fit(self):
        self.stacking.fit(self.train_dataset)

        self.assertEqual(self.stacking.predictions_dataset.shape()[0], self.train_dataset.shape()[0])
        self.assertEqual(len(self.stacking.models), self.stacking.predictions_dataset.shape()[1])


    def test_predict(self):
        self.stacking.fit(self.train_dataset)
        
        predictions = self.stacking.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])
    

    def test_score(self):
        self.stacking.fit(self.train_dataset)
        
        accuracy_ = self.stacking.score(self.test_dataset)
        expected_accuracy = accuracy(self.test_dataset.y, self.stacking.predict(self.test_dataset))

        self.assertEqual(round(accuracy_, 2), round(expected_accuracy, 2))
        print(f"\nObtained accuracy: {(round(accuracy_, 2))}\nExpected accuracy: {round(expected_accuracy, 2)}")